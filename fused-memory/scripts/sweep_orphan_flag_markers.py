#!/usr/bin/env python3
"""Sweep: detect and optionally delete dead-weight stage1_flag_marker records in
Mem0/Qdrant — records missing the ``kind='stage1_flag_marker'`` metadata key
(task-1659 orphans), lacking a usable ``task_id`` (task-2108 orphans), stale by age
(task-1944 precedent), or referencing only terminal tasks (task-2103/2150 precedent).

Task 2596 background
---------------------
Task 2406 retired the Mem0 marker WRITE path entirely — ``flag_dedup.dedup_flags``
now persists markers only to the ``recon_ledger`` SQLite table. Task 2228 (W5-κ)
then deleted the two Mem0 sweeps (``_sweep_stale_flag_markers``,
``_sweep_terminal_task_flag_markers``) that used to drain the legacy Mem0 marker
population, since the ledger's own ``gc()`` pass reaps ledger rows directly. Nothing
was left to drain the pre-2406 Mem0 records, which are pure dead weight (nothing
reads them — see ``find_stale_markers``/``find_terminal_task_markers`` docstrings).
This script's ``find_stale_markers`` and ``find_terminal_task_markers`` restore
those two sweeps' semantics here, as a standalone, deterministic, exit-code-driven
tool usable as a ``task_kind='deterministic'`` ``before_done.script`` (see
``backlog_verdict`` / ``--check``).

Original background (task-1659/2108)
-------------------------------------

Background
----------
Prior to task-1659, ``flag_dedup._write_and_confirm_marker`` wrote markers with
``metadata.source='stage1_flag_marker'`` but omitted ``metadata.kind``.  Dual-filter
queries keyed on *both* source and kind silently under-count those markers.  At the
time, Fix (1) in task-1659 added ``kind`` to every new write; this script was Fix (2):
a one-time sweep that removed the 6 pre-existing orphans so the counts converged
immediately.  (Task 2406 has since retired the write path Fix (1) touched — see
"Task 2596 background" above; there is no longer a new write for it to apply to.)

Deletion vs backfill
--------------------
Mem0/Qdrant now exposes an in-place payload-update primitive: task 3088 shipped
``MemoryService.update_memory`` (``services/memory_service.py:3947``) over
``Mem0Backend.set_payload`` (``backends/mem0_client.py:343``), a genuine
server-side partial merge that preserves the Qdrant point id, ``created_at``,
and every unnamed sibling key. The old "no payload-update primitive on this
path" objection no longer applies, so this script's choice to delete rather
than backfill ``kind`` in place is no longer forced by a missing capability.

Orphan markers are still deleted, not backfilled, for a stronger reason:
nothing reads them. Task 2406 retired the Mem0 marker write path —
``flag_dedup`` persists markers only to the ``recon_ledger`` SQLite table, and
its module docstring (``reconciliation/flag_dedup.py:127-131``) states
plainly: "Reads in this module NEVER consult Mem0 — the ledger is the sole
read source." The write path is doubly closed too: there is no ``add_memory``
call left in ``flag_dedup`` for markers, and a server-side gate
(``server/tools.py:1710-1725``) independently rejects any ``recon-stage-*``
write whose metadata carries ``source`` or ``kind`` equal to
``'stage1_flag_marker'``. So a backfilled ``kind`` on one of these orphans
would be consulted by nothing — it would restore zero dedup capability,
because the population it would join has no live reader left, only a deleter
(see "Task 2596 background" above).

Deletion here is therefore permanent, not self-healing: no code path rewrites
a Mem0 marker on a later MISS cycle any more (that behavior existed before
task 2406 and does not exist now). An operator running ``--apply`` should
read this as an irreversible delete of dead records, not as a correction a
later cycle will reapply.

Taskless markers (task 2108)
-----------------------------
In addition to the missing-``kind`` orphans above, this sweep also purges
stage1_flag_marker records that carry a valid ``kind`` but lack a usable
``task_id`` (missing key, ``None``, or ``''``) — see ``find_taskless_markers``.
This is safe for the same reason as the missing-``kind`` orphans above:
nothing reads the Mem0 marker population at all (see "Deletion vs backfill"),
so a taskless marker is pure dead weight regardless of whether it also
carries a ``kind``. It cannot collapse a repeat flag, cannot suppress
re-escalation, and Stage 2 never sweeps it either. Deleting it loses zero
dedup capability.

Enumeration strategy
--------------------
Markers are enumerated via ``get_memories_by_metadata(filters={'source':MARKER_SOURCE})``,
which performs a deterministic Qdrant payload-filter scroll — NOT semantic search.
Semantic top-N silently drops low-similarity markers (the documented failure mode in
``_query_stage2_flags``), making it unsuitable for exhaustive enumeration.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/sweep_orphan_flag_markers.py

  # Commit the deletions.
  python scripts/sweep_orphan_flag_markers.py --apply

  # Override the target project (default: dark_factory).
  python scripts/sweep_orphan_flag_markers.py --apply --project-id my_project
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

# ---------------------------------------------------------------------------
# Module-level constants
# Cross-reference: payload contract defined in
#   fused_memory.reconciliation.flag_dedup._write_and_confirm_marker
#   (task-1659 adds kind to metadata dict alongside source).
# ---------------------------------------------------------------------------

MARKER_SOURCE: str = 'stage1_flag_marker'
MARKER_KIND: str = 'stage1_flag_marker'

logger = logging.getLogger('sweep_orphan_flag_markers')


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------

def find_orphan_markers(members: list[dict]) -> list[dict]:
    """Return members whose metadata lacks ``kind == MARKER_KIND``.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.

    Returns:
        Subset of *members* for which ``metadata.get('kind') != MARKER_KIND``.
        Order is preserved.  An all-valid input returns ``[]``.
    """
    return [
        m for m in members
        if (m.get('metadata') or {}).get('kind') != MARKER_KIND
    ]


def find_taskless_markers(members: list[dict]) -> list[dict]:
    """Return members whose metadata lacks a usable ``task_id`` (task 2108).

    Mirrors the ``task_id is None or task_id == ''`` emptiness guard used by
    ``flag_dedup.filter_suppressed`` — a marker is taskless when its
    ``task_id`` is missing, ``None``, or ``''``. Numeric and ``fp:``-hash
    task_ids are valid and are never considered taskless, regardless of
    ``kind``.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.

    Returns:
        Subset of *members* for which ``metadata.get('task_id')`` is missing,
        ``None``, or ``''``. Order is preserved. An all-valid input returns ``[]``.
    """
    return [
        m for m in members
        if (m.get('metadata') or {}).get('task_id') in (None, '')
    ]


def classify_marker_task_id(tid: Any) -> str:
    """Bucket a marker's ``task_id`` into one of four shapes (task 2596).

    Buckets:
        - ``'numeric'``: a single all-digit string, e.g. ``'2408'``.
        - ``'fp_hash'``: a canonical ``fp:``+32-hex content-fingerprint key
          (per :func:`is_content_fingerprint_task_id`), e.g.
          ``'fp:' + 'a' * 32``.
        - ``'comma_joined'``: two or more comma-separated components that
          are EACH individually all-digit after stripping whitespace, e.g.
          ``'1944,2408'`` or ``'2405, 540'``. AMENDMENT 1: this never
          naive-``isdigit()``s the whole string (which is always False for
          a comma-joined value) — it splits first and validates each
          sub-id, matching the live record a07972e7 shape
          (``task_id='1944,2408'``).
        - ``'null_or_invalid'``: ``None``, ``''``, a non-``str`` value, or
          any string that matches none of the above (e.g. ``'garbage'``,
          a malformed ``'fp:bad'`` variant, or a comma-joined value with a
          non-digit component like ``'12,x'``).

    Pure, sync, no I/O.

    Args:
        tid: The raw ``metadata.get('task_id')`` value (any type).

    Returns:
        One of ``'numeric'``, ``'fp_hash'``, ``'comma_joined'``,
        ``'null_or_invalid'``.
    """
    if not isinstance(tid, str) or not tid:
        return 'null_or_invalid'
    if is_content_fingerprint_task_id(tid):
        return 'fp_hash'
    if tid.isdigit():
        return 'numeric'
    components = tid.split(',')
    if len(components) >= 2 and all(part.strip().isdigit() for part in components):
        return 'comma_joined'
    return 'null_or_invalid'


def _assume_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it is naive; unchanged otherwise.

    Local copy of the convention centralised in
    ``fused_memory.reconciliation.stages.task_knowledge_sync._assume_utc``
    ("naive datetimes from our journal/Mem0 are UTC") — duplicated here
    rather than imported so this script's pure predicates stay decoupled
    from the heavier reconciliation-stage module.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def find_stale_markers(
    members: list[dict],
    now: datetime,
    max_age_days: int = 14,
) -> list[dict]:
    """Return members whose ``created_at`` is strictly older than the cutoff.

    Restores the age-drain semantics of the retired
    ``_sweep_stale_flag_markers`` (task 1944; deleted by task 2228 W5-κ),
    mirroring ``_sweep_stale_persistence_markers``'s cutoff logic: a member
    is stale when ``created_at < now - max_age_days``. Members with a
    missing, ``None``, or unparseable ``created_at`` are always KEPT
    (never returned) — a fail-safe KEEP-on-uncertainty posture shared with
    every other age-based sweep in this codebase.

    ``max_age_days=0`` sets the cutoff to *now* itself, draining every
    dated member strictly older than *now* — the operator lever to force
    the whole dead-weight population toward zero.

    Pure, sync, no I/O.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.
        now: Reference "current time" for the cutoff calculation. Callers
            inject a fixed value in tests; ``main()`` passes
            ``datetime.now(UTC)``.
        max_age_days: Staleness cutoff in days (default 14).

    Returns:
        Subset of *members* that are stale. Order is preserved.
    """
    cutoff = _assume_utc(now) - timedelta(days=max_age_days)
    stale: list[dict] = []
    for m in members:
        raw = m.get('created_at')
        if raw is None:
            continue
        try:
            created_at = _assume_utc(datetime.fromisoformat(raw))
        except (ValueError, TypeError):
            continue
        if created_at < cutoff:
            stale.append(m)
    return stale


def find_undated_markers(members: list[dict]) -> list[dict]:
    """Return members ``find_stale_markers`` can never drain, at any age cutoff.

    A member is "undated" when its ``created_at`` is missing, ``None``, or
    fails ``datetime.fromisoformat`` parsing — exactly the conditions
    ``find_stale_markers`` fail-safe KEEPs, regardless of ``max_age_days``
    (including ``0``). These members set a floor on the residual backlog
    that no amount of age-draining can reach below; ``run()`` surfaces this
    count via ``undated_kept_count`` plus a WARNING log so an operator
    running ``--check --max-backlog 0`` understands why the backlog can
    floor above zero (task 2596 amendment, reviewer_comprehensive #1/#2).

    Pure, sync, no I/O. Deliberately duplicates ``find_stale_markers``'s
    parse-and-skip logic rather than having that function report both sets,
    so its existing return contract (the stale subset only) stays unchanged
    for existing callers/tests.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.

    Returns:
        Subset of *members* whose ``created_at`` is missing, ``None``, or
        unparseable. Order is preserved. An all-dated input returns ``[]``.
    """
    undated: list[dict] = []
    for m in members:
        raw = m.get('created_at')
        if raw is None:
            undated.append(m)
            continue
        try:
            datetime.fromisoformat(raw)
        except (ValueError, TypeError):
            undated.append(m)
    return undated


def find_terminal_task_markers(
    members: list[dict],
    terminal_task_ids: set[str],
) -> list[dict]:
    """Return members whose ``task_id`` references only terminal tasks.

    Restores the terminal-drain semantics of the retired
    ``_sweep_terminal_task_flag_markers`` (task 2103/2150; deleted by task
    2228 W5-κ), routed through :func:`classify_marker_task_id`:

    - ``'numeric'``: returned iff the task_id is a member of
      *terminal_task_ids*.
    - ``'comma_joined'``: returned iff EVERY comma-separated component is a
      member of *terminal_task_ids* (mirrors the retired sweep's
      split-and-all-terminal rule — a single non-terminal component keeps
      the whole marker).
    - ``'fp_hash'`` / ``'null_or_invalid'``: never returned, regardless of
      *terminal_task_ids*.

    An empty *terminal_task_ids* matches nothing (returns ``[]``).

    Pure, sync, no I/O.

    Args:
        members: List of scroll-shaped dicts ``{'id', 'created_at', 'metadata'}``,
            as returned by ``MemoryService.get_memories_by_metadata``.
        terminal_task_ids: Set of task_id strings whose status is terminal
            (e.g. ``{'done', 'cancelled'}`` per
            ``flag_dedup.TERMINAL_STATUSES``). Injected by the caller — this
            function performs no taskmaster I/O itself.

    Returns:
        Subset of *members* referencing only terminal tasks. Order is
        preserved.
    """
    result: list[dict] = []
    for m in members:
        tid = (m.get('metadata') or {}).get('task_id')
        bucket = classify_marker_task_id(tid)
        if bucket == 'numeric':
            if tid in terminal_task_ids:
                result.append(m)
        elif bucket == 'comma_joined' and isinstance(tid, str):
            # isinstance narrows for pyright; classify_marker_task_id only
            # returns 'comma_joined' for str values, so this is always True.
            components = [part.strip() for part in tid.split(',')]
            if all(part in terminal_task_ids for part in components):
                result.append(m)
    return result


# ---------------------------------------------------------------------------
# Async delete
# ---------------------------------------------------------------------------

async def delete_orphan_markers(
    memory_service: Any,
    project_id: str,
    orphans: list[dict],
    *,
    causation_id: str | None = None,
) -> dict:
    """Delete orphan markers from Mem0 (best-effort, never raises).

    Mirrors the posture of ``_enforce_stage2_summary_pool_cap``:
    asyncio.gather with return_exceptions=True, per-item WARNING on failure,
    count only successes.

    Args:
        memory_service: Live (or mock) MemoryService instance.
        project_id: Project scope passed to each delete_memory call.
        orphans: List of orphan member dicts (output of find_orphan_markers).
        causation_id: Optional causation id forwarded to each delete_memory call.

    Returns:
        ``{'deleted': int, 'failed': [ids]}``
    """
    if not orphans:
        return {'deleted': 0, 'failed': []}

    async def _delete_one(orphan: dict):
        return await memory_service.delete_memory(
            memory_id=orphan['id'],
            store='mem0',
            project_id=project_id,
            causation_id=causation_id,
            _source='sweep_orphan_flag_markers',
        )

    results = await asyncio.gather(
        *(_delete_one(o) for o in orphans),
        return_exceptions=True,
    )

    deleted = 0
    failed: list[str] = []
    for orphan, result in zip(orphans, results, strict=False):
        if isinstance(result, BaseException):
            logger.warning(
                'sweep_orphan_flag_markers: failed to delete memory %s: %s',
                orphan['id'], result,
            )
            failed.append(orphan['id'])
        else:
            deleted += 1

    return {'deleted': deleted, 'failed': failed}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run(
    args: Any,
    memory_service: Any,
    *,
    now: datetime | None = None,
    terminal_task_ids: set[str] | None = None,
) -> dict:
    """Enumerate dead-weight markers and optionally delete them.

    The delete set is the id-deduplicated, order-preserving UNION of four
    automatic predicates (task 2596 restores the two age/terminal drains
    deleted by task 2228 W5-κ alongside the pre-existing orphan/taskless
    ones — see module docstring) plus an optional targeted correction list:
        - ``find_orphan_markers``: missing/mismatched ``kind`` (task 1659).
        - ``find_taskless_markers``: missing/empty ``task_id`` (task 2108).
        - ``find_stale_markers``: ``created_at`` older than ``max_age_days``.
        - ``find_terminal_task_markers``: references only terminal tasks.
        - targeted correction: exact member ids named in ``args.delete_ids``
          (task 2596), force-included regardless of the other predicates —
          the deterministic lever for correcting known-mistagged records
          (e.g. a numeric task_id that is still pending, or a comma-joined
          composite id) that no automatic predicate can catch.
    A member matched by more than one predicate/list is deleted exactly once.

    Args:
        args: argparse.Namespace (or SimpleNamespace) with at least:
            - apply (bool): commit deletions if True, dry-run otherwise
            - project_id (str): project to sweep
            - max_age_days (int, optional): forwarded to find_stale_markers
              (default 14 when absent).
            - delete_ids (list[str], optional): exact marker ids to
              force-delete regardless of age/terminal/orphan status
              (default: none). An id absent from the enumerated members is
              silently ignored.
        memory_service: Live (or mock) MemoryService instance.
        now: Reference "current time" for the age cutoff. Defaults to
            ``datetime.now(UTC)``; tests inject a fixed value.
        terminal_task_ids: Task ids whose status is terminal (e.g. from
            ``flag_dedup.TERMINAL_STATUSES``). Injected by the caller — this
            function performs no taskmaster I/O itself. Defaults to an empty
            set, which makes find_terminal_task_markers a no-op (age-only
            sweep).

    Returns:
        JSON-serialisable report dict:
            - dry_run (bool)
            - before (dict with total_source and total_with_kind counts)
            - orphan_count (int): size of the deduplicated union of all
              predicates/lists — the actual number of records deleted (or
              that would be deleted).
            - orphan_ids (list[str])
            - taskless_orphan_count (int): raw len(find_taskless_markers(...)).
              Overlaps with the kind-orphan predicate for members missing
              BOTH kind and task_id, so it does not subtract cleanly from
              orphan_count — see the inline NOTE above its assignment.
            - undated_kept_count (int): raw len(find_undated_markers(...)) —
              diagnostic only, never part of the delete set. Counts
              enumerated members find_stale_markers can never drain
              regardless of max_age_days (missing/unparseable created_at),
              so operators can see why the residual backlog floors above
              zero under --check/--max-backlog (task 2596 amendment).
            - bucket_counts (dict): count of the final union broken down by
              classify_marker_task_id bucket — always all four keys
              ('numeric', 'fp_hash', 'comma_joined', 'null_or_invalid'),
              even when a bucket's count is 0.
            - targeted_correction_ids (list[str]): the subset of
              ``args.delete_ids`` actually found among the enumerated
              members (the found-intersection, not the raw input list).
            - deleted (int, only when apply=True)
            - failed (list[str], only when apply=True)
            - after (dict with counts, only when apply=True)
    """
    project_id: str = getattr(args, 'project_id', 'dark_factory')
    now_dt: datetime = now if now is not None else datetime.now(UTC)
    terminal_ids: set[str] = terminal_task_ids if terminal_task_ids else set()
    max_age_days: int = getattr(args, 'max_age_days', 14)
    delete_ids: set[str] = set(getattr(args, 'delete_ids', None) or [])

    # --- Before counts (deterministic Qdrant payload-filter, not semantic) ---
    source_filter = {'source': MARKER_SOURCE}
    kind_filter = {'source': MARKER_SOURCE, 'kind': MARKER_KIND}

    total_source = await memory_service.count_memories_by_metadata(
        project_id=project_id, filters=source_filter,
    )
    total_with_kind = await memory_service.count_memories_by_metadata(
        project_id=project_id, filters=kind_filter,
    )
    before = {'total_source': total_source, 'total_with_kind': total_with_kind}

    # --- Enumerate via scroll (NOT semantic search) ---
    scroll_limit: int = getattr(args, 'limit', 1000)
    members = await memory_service.get_memories_by_metadata(
        project_id=project_id, filters=source_filter, limit=scroll_limit,
    )
    # Cross-check: if the scroll returned fewer records than the count, enumeration
    # was capped and some markers were silently skipped — log a warning per the
    # no-silent-caps convention so the operator knows the sweep is incomplete.
    if len(members) < total_source:
        logger.warning(
            'sweep_orphan_flag_markers: enumerated %d of %d source markers '
            '(scroll limit=%d) — scroll cap reached; re-run with a higher '
            '--limit value to ensure all orphans are covered.',
            len(members), total_source, scroll_limit,
        )
    # Orphans are the id-deduplicated, order-preserving union of four
    # independent predicates: missing kind (find_orphan_markers), missing
    # task_id (find_taskless_markers, task 2108), age-stale
    # (find_stale_markers, task 2596 restoring task 1944), and
    # terminal-task-referenced (find_terminal_task_markers, task 2596
    # restoring task 2103/2150) — plus the targeted correction list
    # (args.delete_ids, task 2596), which force-includes specific member ids
    # regardless of what the automatic predicates decide. A member caught by
    # more than one predicate/list is deleted exactly once.
    kind_orphans = find_orphan_markers(members)
    taskless = find_taskless_markers(members)
    stale = find_stale_markers(members, now_dt, max_age_days=max_age_days)
    terminal = find_terminal_task_markers(members, terminal_ids)

    # Diagnostic only — never added to the delete set. Surfaces the subset of
    # `members` find_stale_markers can never drain regardless of
    # --max-age-days (task 2596 amendment, reviewer_comprehensive #1/#2): an
    # operator wiring --check --max-backlog 0 against a population with a
    # nonzero undated_kept_count would otherwise see a perpetual violation
    # with no visibility into why the residual floors above zero.
    undated_kept = find_undated_markers(members)
    if undated_kept:
        logger.warning(
            'sweep_orphan_flag_markers: %d of %d enumerated markers have a '
            'missing/unparseable created_at and are permanently kept by '
            'find_stale_markers regardless of --max-age-days (even 0) — '
            'this sets a floor on the residual backlog that age-draining '
            'alone cannot reach below for --check/--max-backlog. Use '
            '--delete-ids or --terminal-drain to remove them if warranted.',
            len(undated_kept), len(members),
        )
    # Best-effort: an id in delete_ids that doesn't match any enumerated
    # member is simply absent from `targeted` — never a crash.
    targeted = [m for m in members if m['id'] in delete_ids]

    orphans: list[dict] = []
    seen_ids: set = set()
    for m in (*kind_orphans, *taskless, *stale, *terminal, *targeted):
        if m['id'] not in seen_ids:
            seen_ids.add(m['id'])
            orphans.append(m)

    orphan_ids = [o['id'] for o in orphans]
    # The found-intersection of args.delete_ids with the enumerated members
    # (not the raw input list) — order-preserving per `members`.
    targeted_correction_ids = [m['id'] for m in targeted]

    # Per-bucket counts over the final union (task 2596): makes the sweep's
    # action observable by task_id shape without changing orphan_count's
    # meaning (still the total delete-set size).
    bucket_counts: dict[str, int] = {
        'numeric': 0, 'fp_hash': 0, 'comma_joined': 0, 'null_or_invalid': 0,
    }
    for o in orphans:
        tid = (o.get('metadata') or {}).get('task_id')
        bucket_counts[classify_marker_task_id(tid)] += 1

    # NOTE: taskless_orphan_count is the raw size of the taskless predicate
    # (find_taskless_markers), not a "taskless-only" diagnostic — it includes
    # members that are ALSO kind-orphans (e.g. a member missing both kind and
    # task_id). It therefore does not subtract from / sum cleanly against
    # orphan_count to reconstruct the union; it exists purely to make the
    # taskless predicate's contribution to the sweep observable.
    report: dict = {
        'dry_run': not args.apply,
        'before': before,
        'orphan_count': len(orphans),
        'orphan_ids': orphan_ids,
        'taskless_orphan_count': len(taskless),
        'undated_kept_count': len(undated_kept),
        'bucket_counts': bucket_counts,
        'targeted_correction_ids': targeted_correction_ids,
    }

    if args.apply:
        delete_result = await delete_orphan_markers(
            memory_service, project_id, orphans,
        )
        report['deleted'] = delete_result['deleted']
        report['failed'] = delete_result['failed']

        # After counts
        after_source = await memory_service.count_memories_by_metadata(
            project_id=project_id, filters=source_filter,
        )
        after_kind = await memory_service.count_memories_by_metadata(
            project_id=project_id, filters=kind_filter,
        )
        report['after'] = {'total_source': after_source, 'total_with_kind': after_kind}

    return report


# ---------------------------------------------------------------------------
# Deterministic exit-code predicate
# ---------------------------------------------------------------------------

def backlog_verdict(after_total_source: int, max_backlog: int) -> int:
    """Deterministic exit-code predicate: does the residual backlog hold?

    Mirrors ``scripts/check_merge_flakiness.sh``'s exit-code-only contract
    (the orchestrator reads the exit code only, never stdout) so this script
    is directly usable as a ``task_kind='deterministic'``
    ``before_done.script`` predicate (``--apply --check --max-backlog N``).

    Pure, sync, no I/O.

    Args:
        after_total_source: Residual stage1_flag_marker count — the
            report's ``after.total_source`` on ``--apply``, or
            ``before.total_source`` for a dry-run/``--check``-only
            invocation.
        max_backlog: Ceiling the residual count must not exceed.

    Returns:
        ``0`` if ``after_total_source <= max_backlog`` (holds), else ``1``
        (violated).
    """
    return 0 if after_total_source <= max_backlog else 1


def _resolve_check_exit_code(report: dict, max_backlog: int) -> int:
    """Resolve --check's exit code from a sweep report.

    Extracted from :func:`main` (task 2596 amendment, reviewer_comprehensive
    #1) so the report -> exit-code wiring is unit-testable without any live
    I/O: uses ``report['after']['total_source']`` when an ``'after'`` key is
    present (an ``--apply`` run), falling back to
    ``report['before']['total_source']`` otherwise (a dry-run/``--check``-only
    invocation, which never populates ``'after'``).

    Pure, sync, no I/O.

    Args:
        report: The dict returned by :func:`run`.
        max_backlog: Ceiling forwarded to :func:`backlog_verdict`.

    Returns:
        ``0`` if the resolved count holds, else ``1`` — see
        :func:`backlog_verdict`.
    """
    after = report.get('after', report['before'])
    return backlog_verdict(after['total_source'], max_backlog)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _split_comma_ids(raw: str) -> list[str]:
    """argparse ``type=`` callback for ``--delete-ids``.

    Splits *raw* on ``','`` and strips surrounding whitespace from each
    component, dropping any empty components. Pure, sync, no I/O.

    Args:
        raw: Raw ``--delete-ids`` CLI value, e.g. ``'eb92453f, a07972e7'``.

    Returns:
        List of stripped, non-empty id strings, e.g.
        ``['eb92453f', 'a07972e7']``.
    """
    return [part.strip() for part in raw.split(',') if part.strip()]


def _non_negative_int(raw: str, *, flag_name: str) -> int:
    """argparse ``type=`` callback rejecting negative ints, shared by
    ``--max-age-days`` and ``--max-backlog`` (task 2596 amendment,
    reviewer_comprehensive #1).

    Both flags silently misbehave rather than erroring on a negative value:
    ``find_stale_markers`` computes its cutoff as
    ``now - timedelta(days=max_age_days)``, so a negative ``max_age_days``
    pushes the cutoff into the FUTURE and drains every dated member (not
    just stale ones); ``backlog_verdict`` compares ``after_total_source``
    against ``max_backlog`` with ``<=``, so a negative ``max_backlog`` makes
    ANY residual count a violation and a ``--check`` predicate wired with a
    typo'd negative ceiling escalates forever with no explanation. Both are
    the same footgun class, reached by what reads as a typo rather than a
    deliberate value. This script's ``--apply`` path performs irreversible
    deletes and ``--check`` gates deterministic-task completion, so the
    footgun is rejected outright rather than silently honoured.

    Bind ``flag_name`` per call site via :func:`functools.partial` so the
    raised error names the actual flag that failed to parse.

    Args:
        raw: Raw CLI value for the bound flag.
        flag_name: The CLI flag name to cite in the error message, e.g.
            ``'--max-age-days'`` or ``'--max-backlog'``.

    Returns:
        The parsed non-negative int.

    Raises:
        argparse.ArgumentTypeError: If *raw* does not parse as an int, or
            parses to a negative value. argparse turns this into a
            ``parser.error()`` call (exit code 2), matching its handling of
            any other malformed ``type=int`` argument.
    """
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f'invalid int value: {raw!r}') from None
    if value < 0:
        raise argparse.ArgumentTypeError(f'{flag_name} must be >= 0 (got {value}).')
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Build the sweep's argparse parser.

    Factored out of :func:`main` (task 2596) so the CLI surface — including
    the new ``--max-age-days``, ``--delete-ids``, ``--terminal-drain``,
    ``--check``, and ``--max-backlog`` flags — is testable without any live
    I/O.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Detect (and optionally delete) dead-weight stage1_flag_marker '
            'records: missing kind/task_id, stale by age, referencing only '
            'terminal tasks, or explicitly targeted for correction.'
        ),
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help=(
            'Commit deletions of dead-weight stage1_flag_marker records '
            '(default: dry-run only).'
        ),
    )
    parser.add_argument(
        '--project-id', dest='project_id', default='dark_factory',
        help='Project to sweep (default: dark_factory).',
    )
    parser.add_argument(
        '--limit', type=int, default=1000,
        help='Maximum records to enumerate per scroll (default: 1000). '
             'Increase if count_memories_by_metadata shows >1000 source markers.',
    )
    parser.add_argument(
        '--max-age-days', dest='max_age_days',
        type=partial(_non_negative_int, flag_name='--max-age-days'), default=14,
        help='Age cutoff in days for find_stale_markers (default: 14). '
             'Use 0 to drain every dated dead-weight record. Negative '
             'values are rejected (they would push the cutoff into the '
             'future and drain everything, not just stale records).',
    )
    parser.add_argument(
        '--delete-ids', dest='delete_ids', type=_split_comma_ids, default=[],
        help='Comma-separated marker ids to force-delete regardless of '
             'age/terminal/orphan status (targeted correction; default: none).',
    )
    parser.add_argument(
        '--terminal-drain', dest='terminal_drain', action='store_true', default=False,
        help='Best-effort resolve terminal-status (done/cancelled) task ids '
             'via the configured task backend and additionally sweep markers '
             'that reference only such tasks (default: off, age-only sweep — '
             'no taskmaster dependency unless this flag is passed).',
    )
    parser.add_argument(
        '--check', action='store_true', default=False,
        help="Exit 0 if the residual backlog is within --max-backlog, else "
             "1 — usable as a task_kind='deterministic' before_done "
             'predicate (mirrors scripts/check_merge_flakiness.sh).',
    )
    parser.add_argument(
        '--max-backlog', dest='max_backlog',
        type=partial(_non_negative_int, flag_name='--max-backlog'), default=0,
        help=(
            'Residual stage1_flag_marker ceiling checked by --check '
            '(default: 0). Negative values are rejected (a negative '
            'ceiling reached by typo would make backlog_verdict violate '
            'on any residual, forever, with no explanation). A '
            'before_done predicate wired with the default 0 may never be '
            'satisfiable if the population has a nonzero '
            "undated_kept_count (see run()'s report and WARNING log) — "
            'set --max-backlog to at least that count, or run '
            '--delete-ids/--terminal-drain first to clear it.'
        ),
    )
    return parser


async def _resolve_terminal_task_ids() -> set[str]:
    """Best-effort resolve terminal-status task ids for ``--terminal-drain``.

    Mirrors
    ``fused_memory.reconciliation.stages.task_knowledge_sync._resolve_terminal_task_ids``'s
    fail-safe posture: any failure (task backend not configured, backend
    error) degrades to an empty set — the age-only sweep — rather than
    raising, logged at WARNING. Only called when ``--terminal-drain`` is
    passed, so the default run path has no taskmaster dependency at all.

    Returns:
        Set of task_id strings whose status is terminal (``done`` or
        ``cancelled`` per ``shared.task_statuses.TERMINAL``); empty set on
        any failure or unconfigured taskmaster.
    """
    try:
        from shared.task_statuses import TERMINAL as TERMINAL_STATUSES  # noqa: PLC0415

        from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend  # noqa: PLC0415
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        config = FusedMemoryConfig()
        if config.taskmaster is None:
            return set()
        backend = SqliteTaskBackend(config.taskmaster)
        await backend.start()
        try:
            statuses = await backend.get_statuses(config.taskmaster.project_root)
        finally:
            await backend.close()
        return {
            str(tid) for tid, status in statuses.items() if status in TERMINAL_STATUSES
        }
    except Exception:
        # exc_info=True (task 2596 amendment, reviewer_comprehensive #3): a
        # genuine wiring failure (wrong attr, backend import error) must
        # carry a stack trace so it is distinguishable in logs from the
        # unconfigured-taskmaster no-op above, which never reaches here.
        logger.warning(
            'sweep_orphan_flag_markers: --terminal-drain status resolution '
            'failed; falling back to age-only sweep (terminal_task_ids=set()).',
            exc_info=True,
        )
        return set()


def main() -> int:
    """Parse CLI args, build a live MemoryService, and run the sweep."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    parser = _build_parser()
    args = parser.parse_args()

    async def _run_live() -> dict:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        now_dt = datetime.now(UTC)
        terminal_task_ids = (
            await _resolve_terminal_task_ids() if args.terminal_drain else set()
        )
        try:
            await memory.initialize()
            return await run(
                args, memory, now=now_dt, terminal_task_ids=terminal_task_ids,
            )
        finally:
            if hasattr(memory, 'close'):
                await memory.close()

    try:
        report = asyncio.run(_run_live())
    except Exception:
        logger.exception('sweep_orphan_flag_markers: fatal error during sweep')
        return 2

    print(json.dumps(report, indent=2))

    if args.check:
        return _resolve_check_exit_code(report, args.max_backlog)

    return 0


if __name__ == '__main__':
    sys.exit(main())
