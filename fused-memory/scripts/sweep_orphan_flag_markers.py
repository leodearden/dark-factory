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
queries keyed on *both* source and kind silently under-count those markers.  Fix (1) in
task-1659 adds kind to every new write; this script is Fix (2): a one-time sweep to
remove the 6 pre-existing orphans so the counts converge immediately.

Deletion vs backfill
--------------------
Orphan markers are deleted (not updated in place) for two reasons:
 1. Mem0/Qdrant exposes ``delete_memory`` but no payload-update primitive on this path.
 2. stage1_flag_markers are self-healing: a deleted marker is rewritten with both keys
    on the next MISS cycle (at most one extra re-escalation, within the existing
    best-effort-replacement tolerance).

Taskless markers (task 2108)
-----------------------------
In addition to the missing-``kind`` orphans above, this sweep also purges
stage1_flag_marker records that carry a valid ``kind`` but lack a usable
``task_id`` (missing key, ``None``, or ``''``) — see ``find_taskless_markers``.
This is safe for the same self-healing reason as above, plus one more:
``mem0_dedup.find_prior_memories`` filters candidate priors by
``str(meta.get('task_id', '')) != task_id_str`` (see
``fused_memory.reconciliation.flag_dedup``), so a marker without a task_id can
*never* be returned as a dedup prior. It cannot collapse a repeat flag,
cannot suppress re-escalation, and Stage 2 never sweeps it either — it is
pure dead weight from the moment it is written. Deleting it therefore loses
zero dedup capability; if the underlying flag recurs, the next MISS cycle
writes a fresh marker with both ``kind`` and ``task_id`` set.

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
        elif bucket == 'comma_joined':
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
        '--max-age-days', dest='max_age_days', type=int, default=14,
        help='Age cutoff in days for find_stale_markers (default: 14). '
             'Use 0 to drain every dated dead-weight record.',
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
        '--max-backlog', dest='max_backlog', type=int, default=0,
        help='Residual stage1_flag_marker ceiling checked by --check (default: 0).',
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
        logger.warning(
            'sweep_orphan_flag_markers: --terminal-drain status resolution '
            'failed; falling back to age-only sweep (terminal_task_ids=set()).',
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
        after = report.get('after', report['before'])
        return backlog_verdict(after['total_source'], args.max_backlog)

    return 0


if __name__ == '__main__':
    sys.exit(main())
