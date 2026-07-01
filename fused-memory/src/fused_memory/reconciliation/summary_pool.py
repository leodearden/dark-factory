"""Generic per-cycle summary pool-cap enforcement — shared core for Stage 1 and Stage 2.

Extracted from ``reconciliation.stages.task_knowledge_sync``'s Stage-2-specific
``_enforce_stage2_summary_pool_cap`` / ``_pretrim_stage2_summary_pool``
(task 1657 + trim-then-write, task 1831). The trim/pretrim logic itself is
generic — only two identifiers (the ``recon_pool`` tag used for enumeration
and the ``_source`` used for the delete audit trail) differentiate one stage's
pool from another's. Duplicating the ~90-line async GC logic into a second
stage module would create a drift hazard (two copies to keep in sync), so it
lives here once, parametrized, and both stages delegate to it (task 1942).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def _assume_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it is naive; return *dt* unchanged otherwise.

    Centralises the "naive datetimes from our journal/Mem0 are UTC" convention
    for this module. Deliberately duplicated from the sibling copy in
    ``reconciliation.stages.task_knowledge_sync._assume_utc`` rather than
    relocated there — that copy has ~6 existing call sites unrelated to the
    summary pool and relocating it is out of scope for this extraction.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


async def enforce_summary_pool_cap(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    recon_pool: str,
    trim_source: str,
    cap: int,
) -> int:
    """Trim the *recon_pool* pool to at most *cap* members.

    Enumerates all Mem0 memories tagged ``{'recon_pool': recon_pool}`` via
    ``get_memories_by_metadata`` (deterministic Qdrant scroll — NOT semantic
    search), sorts oldest-first by ``created_at``, then deletes the oldest
    ``len - cap`` via parallel ``delete_memory`` calls.

    Best-effort posture (mirrors the original Stage 2
    ``_enforce_stage2_summary_pool_cap``):
    - Enumeration failure → logs WARNING, returns 0, does NOT raise.
    - Individual delete failure → logs WARNING, excluded from count.

    Created_at ordering: uses the ``_assume_utc`` + ``datetime.fromisoformat``
    convention. Members with missing/unparseable ``created_at`` sort LAST
    (treated as newest/kept) so an undatable summary is never preferentially
    deleted.

    Args:
        memory_service: Service with ``get_memories_by_metadata`` and
            ``delete_memory``.
        project_id: Project scope for enumeration and delete calls.
        run_id: Current reconciliation run identifier used as ``causation_id``
            in the audit journal.
        recon_pool: The ``recon_pool`` metadata tag value identifying this pool
            (e.g. ``'stage2_cycle_summary'``, ``'stage1_cycle_summary'``).
        trim_source: The ``_source`` value recorded on each delete for audit
            trail attribution (e.g. ``'stage2_cycle_summary_trim'``).
        cap: Maximum pool size to enforce.

    Returns:
        Number of memories successfully deleted (0 if pool <= cap or on
        enumeration failure).
    """
    try:
        members = await memory_service.get_memories_by_metadata(
            project_id=project_id,
            filters={'recon_pool': recon_pool},
        )
    except Exception:
        logger.warning(
            'reconciliation.enforce_summary_pool_cap: '
            'get_memories_by_metadata failed for project_id=%s recon_pool=%s; skipping trim',
            project_id,
            recon_pool,
            extra={'project_id': project_id, 'run_id': run_id, 'recon_pool': recon_pool},
        )
        return 0

    if len(members) <= cap:
        return 0

    def _sort_key(item: dict) -> tuple:
        raw = item.get('created_at')
        if raw is None:
            return (1, 0)
        try:
            dt = _assume_utc(datetime.fromisoformat(raw))
            return (0, dt)
        except (ValueError, TypeError):
            return (1, 0)

    sorted_members = sorted(members, key=_sort_key)
    to_delete = sorted_members[: len(sorted_members) - cap]

    results = await asyncio.gather(
        *(
            memory_service.delete_memory(
                memory_id=m['id'],
                store='mem0',
                project_id=project_id,
                causation_id=run_id,
                _source=trim_source,
            )
            for m in to_delete
        ),
        return_exceptions=True,
    )

    success_count = 0
    for m, result in zip(to_delete, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                'reconciliation.enforce_summary_pool_cap: '
                'delete failed for memory_id=%s recon_pool=%s; not counted',
                m['id'],
                recon_pool,
                extra={
                    'project_id': project_id,
                    'memory_id': m['id'],
                    'run_id': run_id,
                    'recon_pool': recon_pool,
                },
            )
        else:
            success_count += 1
    return success_count


async def pretrim_summary_pool(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    recon_pool: str,
    trim_source: str,
    cap: int,
) -> int:
    """Pre-trim the *recon_pool* pool to cap-1, reserving one slot.

    Delegates to :func:`enforce_summary_pool_cap` with ``cap=max(cap - 1, 0)``
    so the imminent agent write lands as the cap-th member and can never be a
    trim candidate (trim-then-write ordering, task 1831).

    Must be called BEFORE the agent write (``super().run()`` in the calling
    stage). Post-write pool size transiently reaches cap; it is bounded back
    to cap on the next cycle's pre-trim — no post-write trim is needed.

    Args:
        memory_service: Forwarded to :func:`enforce_summary_pool_cap`.
        project_id: Project scope.
        run_id: Current reconciliation run identifier (audit journal).
        recon_pool: The ``recon_pool`` metadata tag value identifying this pool.
        trim_source: The ``_source`` value recorded on each delete.
        cap: Logical pool cap. Actual trim target is ``max(cap - 1, 0)``.

    Returns:
        Number of memories successfully deleted (0 if pool is already at
        or below cap-1, or on enumeration failure).
    """
    return await enforce_summary_pool_cap(
        memory_service,
        project_id,
        run_id,
        recon_pool=recon_pool,
        trim_source=trim_source,
        cap=max(cap - 1, 0),
    )
