"""Generic per-cycle summary pool-cap enforcement and deterministic ledger
write — shared core for Stage 1 and Stage 2.

Extracted from ``reconciliation.stages.task_knowledge_sync``'s Stage-2-specific
``_enforce_stage2_summary_pool_cap`` / ``_pretrim_stage2_summary_pool``
(task 1657 + trim-then-write, task 1831). The trim logic itself is generic —
only two identifiers (the ``recon_pool`` tag used for enumeration and the
``_source`` used for the delete audit trail) differentiate one stage's pool
from another's. Duplicating the ~90-line async GC logic into a second stage
module would create a drift hazard (two copies to keep in sync), so it lives
here once, parametrized, and both stages delegate to it (task 1942).

Task 2229 (W5-λ) adds :func:`write_cycle_summary`, which supersedes the
LLM-driven nonce/verify/reconstruct self-heal chain that used to live in this
module (task 1572 nonce, task 2366 absence self-heal): Python now writes the
authoritative per-cycle summary directly to the
:class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore` from the
stage's own ``StageReport`` — no LLM turn, no nonce, no absence to self-heal.
The retired mechanisms (and their nonce generator,
``cli_stage_runner.generate_summary_nonce`` / ``build_summary_nonce_section``)
have been deleted now that Stage 1 and Stage 2 ``run()`` are both cut over.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord
from fused_memory.utils.async_utils import gather_collect

logger = logging.getLogger(__name__)

# Retention window for authoritative cycle_summary ledger rows (task 2229).
# The ledger is a control-plane store, not a permanent audit log, and Stage 3
# only ever consumes recent summaries — so rows are given a bounded TTL and
# reaped by the existing ReconLedgerStore.gc() expires_at pass (already run
# each cycle by _gc_recon_markers) rather than kept forever or given bespoke
# cleanup code.
CYCLE_SUMMARY_TTL_DAYS: int = 30


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

    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents the trim sweep from silently converting a shutdown signal
    # into an under-counted deletion tally.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    results = await gather_collect(
        memory_service.delete_memory(
            memory_id=m['id'],
            store='mem0',
            project_id=project_id,
            causation_id=run_id,
            _source=trim_source,
        )
        for m in to_delete
    )

    success_count = 0
    for m, result in zip(to_delete, results, strict=True):
        if isinstance(result, Exception):
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


async def write_cycle_summary(
    memory_service,
    project_id: str,
    report,
    run_id: str,
    *,
    stage: str,
    recon_pool: str,
    trim_source: str,
    cap: int,
    now: datetime | None = None,
) -> bool:
    """Write the authoritative per-cycle ``cycle_summary`` ledger row (task 2229 W5-λ).

    Replaces the LLM-driven per-cycle summary write (nonce + verify/repair/
    reconstruct self-heal, task 2366 and predecessors): Python derives the
    summary directly from *report* — no LLM turn, no dedup race, nothing to
    self-heal — and upserts ONE row into the
    :class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore` keyed
    by ``(project_id, 'cycle_summary', task_id='', flag_type=stage, run_id)``.
    The upsert's ``ON CONFLICT`` on that primary key makes a repeat call for
    the same ``(stage, run_id)`` idempotent — last write wins, row count stays
    1 (boundary test D1).

    Fail-safe no-op when *memory_service* has no ``recon_ledger`` wired
    (mirrors :func:`~fused_memory.reconciliation.stages.task_knowledge_sync._gc_recon_markers`'s
    ``getattr(memory_service, 'recon_ledger', None)`` precedent) — in that
    case neither the mirror nor the trim below run either.

    Once a ledger IS wired, two further best-effort steps run — a Mem0
    mirror (``add_system_record``, human/LLM-searchable) and a pool-cap trim
    (:func:`enforce_summary_pool_cap`, bounds the mirror pool) —
    UNCONDITIONALLY, regardless of whether the authoritative ledger upsert
    itself succeeded: each is independently wrapped in its own try/except
    that logs a WARNING and swallows the failure, so a Mem0 outage can never
    mask (or be masked by) the ledger write's own outcome, and neither can
    ever raise out of this function. The return value reflects ONLY the
    authoritative ledger upsert.

    Args:
        memory_service: Service that may expose a ``recon_ledger``
            (:class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore`)
            attribute. Missing/``None`` => no-op (returns ``False``).
        project_id: Project scope for the ledger row.
        report: The stage's own ``StageReport`` — ``started_at``,
            ``completed_at``, ``items_flagged``, ``stats``, ``llm_calls``, and
            ``tokens_used`` are serialized into ``payload_json`` verbatim (the
            authoritative copy of what this cycle did).
        run_id: Current reconciliation run identifier (identity key).
        stage: The ``stage`` metadata tag value identifying this stage (e.g.
            ``'memory_consolidator'``, ``'task_knowledge_sync'``) — stored as
            the ledger row's ``flag_type`` (the discriminator that keeps Stage
            1 and Stage 2 rows for the same ``run_id`` from colliding).
        recon_pool: Forwarded to the best-effort Mem0 mirror pool-cap trim.
        trim_source: Forwarded to the best-effort Mem0 mirror pool-cap trim.
        cap: Forwarded to the best-effort Mem0 mirror pool-cap trim.
        now: Reference "current time" for ``created_at``/``expires_at``.
            Defaults to ``datetime.now(UTC)``; tests inject a fixed value.
            Normalized via :func:`_assume_utc` and rendered with
            ``.isoformat()`` so the ledger's lexicographic TEXT ``gc()``
            comparison against ``expires_at`` stays correct.

    Returns:
        ``True`` when the authoritative ledger upsert succeeded, ``False``
        otherwise (no ledger wired, or the upsert raised).
    """
    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        return False

    now_dt = _assume_utc(now or datetime.now(UTC))
    written = True
    payload: dict | None = None
    try:
        # Payload/record construction lives inside this try (not just the
        # upsert call) so a malformed report — e.g. a stray None
        # started_at/completed_at — degrades to written=False + WARNING
        # like any other ledger-write failure, rather than raising an
        # unhandled AttributeError out of this fail-safe function.
        payload = {
            'stage': stage,
            'run_id': run_id,
            'started_at': report.started_at.isoformat(),
            'completed_at': report.completed_at.isoformat(),
            'items_flagged_count': len(report.items_flagged),
            'stats': report.stats,
            'llm_calls': report.llm_calls,
            'tokens_used': report.tokens_used,
        }
        record = ReconLedgerRecord(
            project_id=project_id,
            record_kind='cycle_summary',
            task_id='',
            flag_type=stage,
            run_id=run_id,
            payload_json=json.dumps(payload, default=str),
            state='active',
            created_at=now_dt.isoformat(),
            expires_at=(now_dt + timedelta(days=CYCLE_SUMMARY_TTL_DAYS)).isoformat(),
        )
        await ledger.upsert(record)
    except Exception:
        logger.warning(
            'reconciliation.write_cycle_summary: '
            'ledger upsert failed for run_id=%s stage=%s',
            run_id,
            stage,
            exc_info=True,
            extra={'project_id': project_id, 'run_id': run_id, 'stage': stage},
        )
        written = False

    try:
        # payload is None only when construction itself failed above (already
        # logged as a ledger-upsert WARNING) — nothing meaningful to mirror.
        if payload is not None:
            await memory_service.add_system_record(
                content=_build_cycle_summary_mirror_content(run_id, stage, payload),
                project_id=project_id,
                agent_id=f'recon-stage-{stage}',
                category='observations_and_summaries',
                metadata={'kind': 'cycle_summary', 'stage': stage, 'run_id': run_id},
                causation_id=run_id,
                _source='cycle_summary_mirror',
            )
    except Exception:
        logger.warning(
            'reconciliation.write_cycle_summary: '
            'add_system_record mirror failed for run_id=%s stage=%s',
            run_id,
            stage,
            exc_info=True,
            extra={'project_id': project_id, 'run_id': run_id, 'stage': stage},
        )

    try:
        await enforce_summary_pool_cap(
            memory_service,
            project_id,
            run_id,
            recon_pool=recon_pool,
            trim_source=trim_source,
            cap=cap,
        )
    except Exception:
        logger.warning(
            'reconciliation.write_cycle_summary: '
            'enforce_summary_pool_cap failed for run_id=%s stage=%s recon_pool=%s',
            run_id,
            stage,
            recon_pool,
            exc_info=True,
            extra={
                'project_id': project_id,
                'run_id': run_id,
                'stage': stage,
                'recon_pool': recon_pool,
            },
        )

    return written


def _build_cycle_summary_mirror_content(run_id: str, stage: str, payload: dict) -> str:
    """Build the deterministic Mem0 mirror content for *run_id* / *stage*.

    Deterministic — no CSPRNG nonce, no LLM turn, nothing to dedup-defeat
    (unlike the retired :func:`_build_fallback_summary_content`): the mirror
    is a best-effort searchable copy of the ledger's authoritative payload,
    not itself a source of truth, so an occasional Mem0 dedup drop of this
    write is harmless. Embeds ``run_id`` and *stage* verbatim so a semantic
    or metadata-filtered search on either surfaces this record.
    """
    return (
        f'Stage {stage} cycle summary for run_id: {run_id} — '
        f'{payload["items_flagged_count"]} item(s) flagged, '
        f'{payload["llm_calls"]} llm_call(s), {payload["tokens_used"]} token(s) used.'
    )
