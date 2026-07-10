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

from fused_memory.reconciliation.cli_stage_runner import generate_summary_nonce
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
    written = True
    try:
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


# SYNC WARNING (task 2366): near-verbatim duplicated as
# task_knowledge_sync._verify_stage2_summary_written (same triple filter,
# hardcoded to stage='task_knowledge_sync'). That copy was deliberately left
# in place — not refactored to delegate here — to avoid regression risk to a
# proven production path (see this task's plan: "Stage 2 delegation to it is
# a noted follow-up (not in scope, to avoid Stage 2 regression risk)").
# Editing the verify logic here is a signal to check whether the Stage 2
# copy needs the same fix, and vice versa.
async def verify_cycle_summary_written(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    stage: str,
) -> int | None:
    """Best-effort post-write check: count this run's cycle_summary in Mem0.

    Generalizes ``task_knowledge_sync._verify_stage2_summary_written`` (task
    2366) to be stage-agnostic. Counts memories matching the triple filter
    ``{'kind':'cycle_summary','run_id':run_id,'stage':stage}`` via
    ``count_memories_by_metadata`` (deterministic Qdrant count, NOT semantic
    search). Logs a WARNING when count==0 or when the count call raises.

    Best-effort: never raises, never retries. The LLM agent owns its own
    count-verify-and-retry path; this is a redundant harness-side
    observability/self-heal signal surfaced via a caller's
    ``report.stats[...]``.

    Args:
        memory_service: Service with ``count_memories_by_metadata``.
        project_id: Project scope for the count call.
        run_id: Current reconciliation run identifier (used as filter key).
        stage: The ``stage`` metadata tag value identifying this stage
            (e.g. ``'memory_consolidator'``, ``'task_knowledge_sync'``).

    Returns:
        Count returned by ``count_memories_by_metadata`` when the call
        succeeds — 0 means CONFIRMED absent (the count query ran cleanly and
        found nothing). ``None`` when the count call itself raised — a
        transient failure that does NOT confirm absence. This distinction is
        load-bearing: callers must gate reconstruction on ``verified_count ==
        0`` specifically, never on a falsy value, so a transient outage never
        fabricates a reconstruction placeholder for a run whose real summary
        may already exist.
    """
    try:
        count = await memory_service.count_memories_by_metadata(
            project_id=project_id,
            filters={'kind': 'cycle_summary', 'run_id': run_id, 'stage': stage},
        )
    except Exception:
        logger.warning(
            'reconciliation.verify_cycle_summary_written: '
            'count_memories_by_metadata failed for run_id=%s stage=%s; absence NOT '
            'confirmed (transient failure, not treated as count==0)',
            run_id,
            stage,
            extra={'project_id': project_id, 'run_id': run_id, 'stage': stage},
        )
        return None
    if not count:
        logger.warning(
            'reconciliation.verify_cycle_summary_written: '
            'no cycle_summary found for run_id=%s stage=%s after agent write',
            run_id,
            stage,
            extra={'project_id': project_id, 'run_id': run_id, 'stage': stage},
        )
    return count


# SYNC WARNING (task 2366): _extract_response_memory_ids,
# _build_fallback_summary_content, and reconstruct_cycle_summary_stub below
# are near-verbatim duplicated by task_knowledge_sync._extract_response_memory_ids
# / _build_stage2_reconstruction_content / _reconstruct_stage2_summary. Those
# copies were deliberately left in place — not refactored to delegate here —
# to avoid regression risk to a proven production path (see this task's
# plan). Editing any of the three functions below is a signal to check
# whether the Stage 2 copies need the same fix, and vice versa.
def _extract_response_memory_ids(response) -> list:
    """Defensively read ``memory_ids`` from an ``add_memory`` response.

    Production calls return :class:`~fused_memory.models.memory.AddMemoryResponse`
    (attribute access); tests commonly mock ``add_memory`` with a plain
    ``{'memory_ids': [...]}`` dict. Supports both shapes so callers never need
    to know which one they were handed. Mirrors
    ``task_knowledge_sync._extract_response_memory_ids``.

    Returns:
        The ``memory_ids`` list, or ``[]`` if absent/falsy on either shape.
    """
    if isinstance(response, dict):
        return response.get('memory_ids') or []
    return getattr(response, 'memory_ids', None) or []


# Maps a `stage` metadata tag value to the nonce label generate_summary_nonce
# expects (task 1590's 'STAGE1' / 'STAGE2' convention — see
# cli_stage_runner.py). Keeps _build_fallback_summary_content's content
# genuinely stage-agnostic: a Stage 2 stub must lead with a STAGE2_-prefixed
# nonce, not a hardcoded STAGE1_ one, so the label stays an honest signal of
# the stub's origin (task 2366 amendment). An unmapped stage falls back to
# its own uppercased name rather than raising, matching this module's
# best-effort posture.
_NONCE_PREFIX_BY_STAGE = {
    'memory_consolidator': 'STAGE1',
    'task_knowledge_sync': 'STAGE2',
}


def _nonce_prefix_for_stage(stage: str) -> str:
    """Return the ``generate_summary_nonce`` prefix label for *stage*."""
    return _NONCE_PREFIX_BY_STAGE.get(stage, stage.upper())


def _build_fallback_summary_content(run_id: str, stage: str, attempt: int) -> str:
    """Build the deterministic fallback-stub content for *run_id* / *stage*.

    Leads with a fresh ``generate_summary_nonce(<prefix>)`` line — the same
    CSPRNG dedup-defeat primitive the LLM per-cycle-summary path uses (task
    1572/1590) — so repeat calls (the one-shot retry in
    :func:`reconstruct_cycle_summary_stub`) never collide on Mem0's ~0.92
    cosine-similarity dedup threshold. The prefix is derived from *stage* via
    :func:`_nonce_prefix_for_stage` (task 2366 amendment), so the label
    matches the stage the stub claims instead of being hardcoded to Stage 1's.
    *attempt* does not otherwise affect the content — distinctness between
    attempts comes entirely from the fresh nonce — it is accepted purely for
    readability/parity with the Stage 2 template this generalizes.

    The body embeds ``run_id: <run_id>`` verbatim so a semantic search on the
    run_id also matches this stub (Path-1 self-heal), in addition to the
    metadata triple-filter (Path-2).

    Args:
        run_id: Current reconciliation run identifier.
        stage: The ``stage`` metadata tag value identifying this stage.
        attempt: 1 for the first write, 2 for the dedup retry (informational).

    Returns:
        Fallback stub content, always starting with a fresh nonce line.
    """
    nonce = generate_summary_nonce(_nonce_prefix_for_stage(stage))
    return (
        f'{nonce}\n'
        f'Stage {stage} cycle summary (deterministic harness fallback stub) for '
        f'run_id: {run_id} — attempt {attempt}. The LLM session exited without '
        'writing its per-cycle summary; this stub was reconstructed by the '
        'harness self-heal and reflects no substantive mutations.'
    )


async def reconstruct_cycle_summary_stub(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    stage: str,
    recon_pool: str,
    reconstruct_source: str,
) -> int:
    """Write ONE dedup-resilient fallback cycle_summary stub for *run_id*.

    Generalizes ``task_knowledge_sync._reconstruct_stage2_summary`` (task
    2366) to be stage-agnostic. Called when :func:`verify_cycle_summary_written`
    has CONFIRMED (``verified_count == 0``) that no cycle_summary exists for
    this run — closing the reliability gap left by a prompt-driven summary
    write that a stage's LLM session can skip when it exhausts its
    turn/token/context budget before reaching its final write.

    Writes via ``add_memory`` tagged with the four canonical identity keys
    (``kind='cycle_summary'``, ``stage``, ``run_id``, ``recon_pool``) plus
    ``reconstructed=True``, and ``_source=reconstruct_source`` so the write is
    auditable and distinguishable from a genuine agent write. The content
    (:func:`_build_fallback_summary_content`) leads with a CSPRNG nonce to
    defeat Mem0's dedup.

    Because the root cause can itself be a dedup drop, the write is
    dedup-resilient: the response's ``memory_ids`` is inspected
    (:func:`_extract_response_memory_ids`), and if empty (a dedup no-op) the
    write is retried exactly ONCE with a fresh nonce (metadata unchanged).
    This mirrors the LLM prompt's own retry_nonce pattern without risking an
    unbounded loop.

    Best-effort: never raises. Any ``add_memory`` exception (first attempt or
    retry) logs a WARNING and returns 0 — a memory-store outage degrades this
    self-heal gracefully instead of failing the whole reconciliation cycle.

    Args:
        memory_service: Service with ``add_memory``.
        project_id: Project scope for the write.
        run_id: Current reconciliation run identifier — both the metadata
            identity key and the ``causation_id`` for the write.
        stage: The ``stage`` metadata tag value identifying this stage.
        recon_pool: The ``recon_pool`` metadata tag value identifying this
            stage's cycle-summary pool — so the stub participates in the
            next cycle's pool-cap trim exactly like a real summary.
        reconstruct_source: The ``_source`` value recorded on the write for
            audit trail attribution.

    Returns:
        1 if a stub was successfully written (first attempt or retry — see
        :func:`_extract_response_memory_ids`), else 0 (dedup-dropped twice,
        or ``add_memory`` raised).
    """
    metadata = {
        'kind': 'cycle_summary',
        'stage': stage,
        'run_id': run_id,
        'recon_pool': recon_pool,
        'reconstructed': True,
    }
    try:
        response = await memory_service.add_memory(
            content=_build_fallback_summary_content(run_id, stage, 1),
            category='observations_and_summaries',
            project_id=project_id,
            metadata=metadata,
            causation_id=run_id,
            _source=reconstruct_source,
        )
        if _extract_response_memory_ids(response):
            return 1

        # Dedup no-op (Mem0 ~0.92 cosine similarity) — retry once with a fresh
        # nonce so the leading content shifts past the threshold.
        response = await memory_service.add_memory(
            content=_build_fallback_summary_content(run_id, stage, 2),
            category='observations_and_summaries',
            project_id=project_id,
            metadata=metadata,
            causation_id=run_id,
            _source=reconstruct_source,
        )
        if _extract_response_memory_ids(response):
            return 1
    except Exception:
        logger.warning(
            'reconciliation.reconstruct_cycle_summary_stub: '
            'add_memory failed for run_id=%s stage=%s; skipping reconstruction',
            run_id,
            stage,
            exc_info=True,
            extra={'project_id': project_id, 'run_id': run_id, 'stage': stage},
        )
        return 0

    logger.warning(
        'reconciliation.reconstruct_cycle_summary_stub: '
        'reconstruction write dedup-dropped twice for run_id=%s stage=%s; giving up',
        run_id,
        stage,
        extra={'project_id': project_id, 'run_id': run_id, 'stage': stage},
    )
    return 0
