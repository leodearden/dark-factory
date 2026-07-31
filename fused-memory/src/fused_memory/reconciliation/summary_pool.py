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

from fused_memory.reconciliation.mem0_tombstone import record_mem0_deletion_tombstone
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord
from fused_memory.reconciliation.recon_pool_map import (
    CYCLE_SUMMARY_KIND,
)
from fused_memory.reconciliation.recon_pool_map import (
    CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP as _CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP,
)
from fused_memory.reconciliation.recon_pool_map import (
    CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE as _CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE,
)
from fused_memory.utils.async_utils import gather_collect

logger = logging.getLogger(__name__)

# Retention window for authoritative cycle_summary ledger rows (task 2229).
# The ledger is a control-plane store, not a permanent audit log, and Stage 3
# only ever consumes recent summaries — so rows are given a bounded TTL and
# reaped by the existing ReconLedgerStore.gc() expires_at pass (already run
# each cycle by _gc_recon_markers) rather than kept forever or given bespoke
# cleanup code.
CYCLE_SUMMARY_TTL_DAYS: int = 30

# record_type vocabulary for cycle_summary Mem0 writes (task 2468). There are
# two distinct writers of kind='cycle_summary': this module's deterministic,
# terse, auto-generated Mem0 mirror of the authoritative ledger row
# (LEDGER_STAMP, written unconditionally below), and the LLM-authored
# reconstruction/self-heal cycle_summary write in
# ``reconciliation.prompts.stage2`` (NARRATIVE).
#
# record_type was write-only as of task 2468: no reader (dedup/near-duplicate
# tooling, Path-2 verification, pool-cap trim) filtered on it — the fix that
# actually stopped the double-write was the removed normal-flow LLM
# instruction (recon_self_model.py), not this discriminator.
#
# Task 3041 gives LEDGER_STAMP its first two real readers: this module's own
# record_type-aware eviction order in enforce_summary_pool_cap below, and
# reconciliation.mem0_tombstone.is_protected_mirror_record. Because
# mem0_tombstone is imported BY this module (for the trim-path tombstone
# write), it cannot import back — so the literals now live in the leaf
# recon_pool_map alongside the pool names, single-sourced for both readers,
# and are re-exported here under their historical names. See that module for
# why (task 3041 amendment pass: they were previously duplicated with nothing
# pinning the copies equal, so an edit to one side would silently disable half
# the protected-mirror guard).
#
# NARRATIVE still has no Python consumer; keeping the prompt-side literal
# (prompts/stage2.py, recon_self_model.py) in sync remains a reviewed
# invariant rather than an enforced one.
CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP: str = _CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP
CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE: str = _CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE

# metadata.kind identifying a cycle_summary record. Written by
# write_cycle_summary's mirror below, keyed on by
# services.memory_service._apply_cycle_summary_metadata_tagging, and now also
# the enumeration constraint for enforce_summary_pool_cap (task 3041).
# Single-sourced in recon_pool_map for the same reason as the record_types
# above — mem0_tombstone.is_protected_mirror_record needs the same literal.
_KIND_CYCLE_SUMMARY: str = CYCLE_SUMMARY_KIND

# Explicit scroll bound for the pool enumeration below (task 3041). Same value
# as get_memories_by_metadata's own default, which this previously relied on
# implicitly. Named and passed explicitly because it is a CORRECTNESS bound,
# not a performance knob: a pool that reached this size would enumerate as a
# partial view, and Qdrant scroll order is not guaranteed oldest-first, so the
# trim would then keep an arbitrary subset while still reporting success.
# Reaching it is not expected — this trim is what holds the pool at `cap`
# every cycle — so hitting it means something upstream is already wrong, and
# enforce_summary_pool_cap says so out loud rather than degrading silently.
SUMMARY_POOL_SCROLL_LIMIT: int = 1000


def _assume_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it is naive; return *dt* unchanged otherwise.

    Centralises the "naive datetimes from our journal/Mem0 are UTC" convention
    for this module. Deliberately duplicated from the sibling copy in
    ``reconciliation.stages.task_knowledge_sync._assume_utc`` rather than
    relocated there — that copy has ~6 existing call sites unrelated to the
    summary pool and relocating it is out of scope for this extraction.
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


async def _warn_on_untrimmable_pool_residue(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    recon_pool: str,
    trimmable: int,
) -> int | None:
    """Warn when the pool holds records this trim can no longer see.

    :func:`enforce_summary_pool_cap` enumerates on
    ``{'recon_pool': ..., 'kind': 'cycle_summary'}``. The ``kind`` constraint
    is what stops a mis-tagged record from evicting a real mirror — but it
    also means a record carrying ``recon_pool`` WITHOUT
    ``kind='cycle_summary'`` is no longer enumerated, so it is never evicted
    either. It is not covered by the protected-mirror guard, and no other
    collector claims it. Before the ``kind`` constraint such a record was
    trimmed; after it, it would accumulate with ZERO signal — which is the
    unbounded pool growth tasks 1657/1831/2229 built this trim to prevent,
    reintroduced in a shape nothing reports (reviewer finding robustness,
    task 3041 amendment pass).

    That shape is realistic, not theoretical: cycle_summary metadata is
    LLM-supplied on the narrative write path, and
    ``_apply_cycle_summary_metadata_tagging`` backfills ``run_id`` precisely
    BECAUSE prompt compliance is not guaranteed — a write that lands
    ``recon_pool`` (or has it auto-stamped from ``metadata.stage``) while
    dropping ``kind`` produces exactly this residue.

    So the narrowed delete filter stays and the pool gets an observability
    backstop instead: one ``count_memories_by_metadata`` on the ``recon_pool``
    tag ALONE. Anything in that count beyond what the narrowed enumeration
    returned is untrimmable residue, and gets one WARNING naming the size.

    Diagnostic-only and fully fail-safe — this must never change what the trim
    does, only what it says:

    - a ``memory_service`` without ``count_memories_by_metadata`` (the shape
      several inline test fakes have) is skipped silently;
    - a non-int result (an unspecced mock) is skipped silently;
    - a raising count logs one WARNING and returns, since a blind backstop is
      itself worth knowing about, but the trim proceeds regardless.

    Returns:
        The residue size when one was computed, ``0`` when the pool is clean,
        or ``None`` when the check could not run.
    """
    counter = getattr(memory_service, 'count_memories_by_metadata', None)
    if counter is None:
        return None
    try:
        total = await counter(project_id=project_id, filters={'recon_pool': recon_pool})
    except Exception:
        logger.warning(
            'reconciliation.enforce_summary_pool_cap: '
            'untrimmable-residue count failed for project_id=%s recon_pool=%s; '
            'the trim itself is unaffected, but mis-tagged pool residue is now unreported',
            project_id,
            recon_pool,
            exc_info=True,
            extra={'project_id': project_id, 'run_id': run_id, 'recon_pool': recon_pool},
        )
        return None
    if not isinstance(total, int) or isinstance(total, bool):
        return None

    residue = total - trimmable
    if residue <= 0:
        return 0

    logger.warning(
        'reconciliation.enforce_summary_pool_cap: '
        '%d record(s) tagged recon_pool=%s are NOT kind=%s in project_id=%s — '
        'they are invisible to this trim and no other collector reaps them; '
        'pool total=%d, trimmable=%d',
        residue,
        recon_pool,
        _KIND_CYCLE_SUMMARY,
        project_id,
        total,
        trimmable,
        extra={
            'project_id': project_id,
            'run_id': run_id,
            'recon_pool': recon_pool,
            'pool_total': total,
            'trimmable': trimmable,
            'untrimmable_residue': residue,
        },
    )
    return residue


async def enforce_summary_pool_cap(
    memory_service,
    project_id: str,
    run_id: str,
    *,
    recon_pool: str,
    trim_source: str,
    cap: int,
) -> int:
    """Trim the ``kind='cycle_summary'`` members of the *recon_pool* pool to at most *cap*.

    Enumerates all Mem0 memories tagged
    ``{'recon_pool': recon_pool, 'kind': 'cycle_summary'}`` via
    ``get_memories_by_metadata`` (deterministic Qdrant scroll — NOT semantic
    search), sorts eviction-order-first, then deletes the first ``len - cap``
    via parallel ``delete_memory`` calls.

    **Scope note (task 3041):** despite the generic name this is
    cycle-summary-specific — the ``kind`` constraint below means a record
    tagged with this ``recon_pool`` but NOT ``kind='cycle_summary'`` is
    neither enumerated nor evicted here. Nothing else collects such a record
    either, so it would accumulate silently; instead it is counted and
    reported by :func:`_warn_on_untrimmable_pool_residue`, which runs on every
    call and is diagnostic-only.

    Best-effort posture (mirrors the original Stage 2
    ``_enforce_stage2_summary_pool_cap``):
    - Enumeration failure → logs WARNING, returns 0, does NOT raise.
    - Individual delete failure → logs WARNING, excluded from count.

    Every CONFIRMED-successful eviction leaves a queryable tombstone via
    :func:`~fused_memory.reconciliation.mem0_tombstone.record_mem0_deletion_tombstone`
    (task 3041), naming *trim_source* as the deleter and *run_id* as the
    deleting run — deliberately distinct from the victim's own
    ``metadata['run_id']``, which is also recorded. Written from the success
    branch ONLY: a tombstone must never claim a record that is still alive.

    **Retention contract (task 3041).** Eviction order is
    ``(is_ledger_stamp, has_parseable_created_at, created_at)``:

    1. ``record_type='narrative'`` records are evicted BEFORE any
       ``record_type='ledger_stamp'`` record. The ledger_stamp mirror is the
       deterministic copy an auditor correlates against
       :meth:`~fused_memory.services.memory_service.MemoryService.get_cycle_summary_presence`;
       the narrative is the disposable LLM-authored duplicate that task 2468
       already tried to suppress. Letting a narrative evict a ledger_stamp is
       what made all three of run 84eae9bd's anchors vanish together (recon
       gate 165 / esc-165-1).
    2. Within a class, oldest-first by ``created_at`` (``_assume_utc`` +
       ``datetime.fromisoformat``).
    3. Members with missing/unparseable ``created_at`` sort LAST *within their
       own class* (treated as newest/kept), preserving the pre-existing
       invariant that an undatable summary is never preferentially deleted.
    ``record_type`` is read defensively from ``item.get('metadata', {})`` — a
    member dict with no metadata must not raise.

    Enumeration is bounded by :data:`SUMMARY_POOL_SCROLL_LIMIT`, passed
    explicitly rather than inherited from ``get_memories_by_metadata``'s
    default. With a cap of 2 the bound is unreachable in normal operation —
    this trim is what holds the pool at ``cap`` every cycle — but it is a
    correctness bound rather than a performance knob: at the limit the view is
    potentially PARTIAL, Qdrant scroll order is not guaranteed oldest-first,
    and the survivors would then be an arbitrary subset rather than the newest
    ``cap``. Reaching it therefore logs a WARNING and still trims, instead of
    silently returning a count that reads as "pool trimmed to cap".

    The ``kind`` filter constraint is load-bearing, not decorative:
    ``_apply_cycle_summary_metadata_tagging`` is additive-only and never
    strips a caller-supplied ``recon_pool``, so filtering on ``recon_pool``
    alone would let a mis-tagged non-summary record join this pool and either
    be trimmed by it or evict a real mirror.

    **This pool is cap-bounded BY DESIGN, and that is not a bug.** A mirror
    older than the newest *cap* ledger_stamps IS expected to be evicted — the
    Mem0 mirror is documented as a best-effort searchable copy, and the
    AUTHORITATIVE record is the ``ReconLedgerStore`` ``cycle_summary`` row
    (read via ``get_cycle_summary_presence``), which survives untouched. What
    was actually broken was DISCOVERABILITY: a designed eviction was
    indistinguishable from silent data loss. From task 3041 on, every
    eviction leaves a queryable tombstone. Raising the cap would only move
    the cliff and trade a bounded pool for the unbounded growth tasks
    1657/1831/2229 built this trim to prevent — so do NOT raise it.

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
            # kind is load-bearing, not decorative (task 3041):
            # _apply_cycle_summary_metadata_tagging is ADDITIVE-only and never
            # strips a caller-supplied recon_pool, so filtering on recon_pool
            # alone would let a mis-tagged non-summary record join this cap-2
            # pool — and then either be trimmed by it or evict a real mirror.
            filters={'recon_pool': recon_pool, 'kind': _KIND_CYCLE_SUMMARY},
            limit=SUMMARY_POOL_SCROLL_LIMIT,
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

    # Diagnostic only — never changes what is deleted, and deliberately runs
    # BEFORE the under-cap early return: mis-tagged residue accumulates
    # whether or not this cycle has anything to trim.
    await _warn_on_untrimmable_pool_residue(
        memory_service,
        project_id,
        run_id,
        recon_pool=recon_pool,
        trimmable=len(members),
    )

    if len(members) >= SUMMARY_POOL_SCROLL_LIMIT:
        # The enumeration may be a PARTIAL view of the pool, so the members
        # this trim is about to sort are not necessarily the whole set and the
        # survivors are not necessarily the newest `cap`. The trim still runs
        # (bounding a runaway pool beats doing nothing), but a pool this size
        # means something upstream is already broken — say so rather than
        # returning a success count that reads as "pool trimmed to cap".
        logger.warning(
            'reconciliation.enforce_summary_pool_cap: '
            'enumeration returned %d members at the scroll limit for '
            'project_id=%s recon_pool=%s — the pool view may be PARTIAL and '
            'the retained members may not be the newest %d; trimming anyway',
            len(members),
            project_id,
            recon_pool,
            cap,
            extra={
                'project_id': project_id,
                'run_id': run_id,
                'recon_pool': recon_pool,
                'member_count': len(members),
            },
        )

    if len(members) <= cap:
        return 0

    def _sort_key(item: dict) -> tuple:
        """Eviction order: narratives first, then oldest-first within a class.

        Three-part key ``(is_ledger_stamp, has_parseable_created_at,
        created_at)`` — ``sorted`` is ascending and the head of the list is
        deleted, so ``False``/``0`` sorts first == is evicted first.
        """
        metadata = item.get('metadata')
        record_type = (
            metadata.get('record_type') if isinstance(metadata, dict) else None
        )
        is_ledger_stamp = record_type == CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP

        raw = item.get('created_at')
        if raw is None:
            return (is_ledger_stamp, 1, 0)
        try:
            dt = _assume_utc(datetime.fromisoformat(raw))
            return (is_ledger_stamp, 0, dt)
        except (ValueError, TypeError):
            return (is_ledger_stamp, 1, 0)

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
    tombstone_writes = []
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
            # THIS is the path that consumed the three run-84eae9bd
            # cycle_summary anchors reported by recon gate 165 / esc-165-1 —
            # a designed cap-2 eviction that left no trace linking the
            # deletion to its victim. The tombstone, NOT a retention change,
            # is the fix for that finding's "no audit trail" signature: the
            # eviction itself was correct and stays.
            #
            # Success branch ONLY: a tombstone must never claim a record that
            # is still alive. record_mem0_deletion_tombstone is internally
            # fail-safe (returns False, never raises); gathering the writes
            # through gather_collect is a second belt so nothing here can
            # raise out of, or alter the count of, this trim.
            tombstone_writes.append(
                record_mem0_deletion_tombstone(
                    memory_service,
                    project_id,
                    m['id'],
                    victim_metadata=m.get('metadata'),
                    victim_created_at=m.get('created_at'),
                    deleter=trim_source,
                    deleting_run_id=run_id,
                )
            )

    if tombstone_writes:
        await gather_collect(tombstone_writes)

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
    remediation: bool = False,
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

    The authoritative ledger write is a no-op when *memory_service* has no
    ``recon_ledger`` wired (mirrors :func:`~fused_memory.reconciliation.stages.task_knowledge_sync._gc_recon_markers`'s
    ``getattr(memory_service, 'recon_ledger', None)`` precedent): the ledger
    upsert is skipped and this function returns ``False``.

    The Mem0 mirror (``add_system_record``, human/LLM-searchable) and the
    pool-cap trim (:func:`enforce_summary_pool_cap`, bounds the mirror pool)
    run UNCONDITIONALLY — regardless of whether a ledger is wired at all, and
    regardless of whether the authoritative ledger upsert itself succeeded.
    Each is independently wrapped in its own try/except that logs a WARNING
    and swallows the failure, so neither a missing ledger nor a Mem0 outage
    can mask (or be masked by) the other's outcome, and neither can ever
    raise out of this function. This matters in practice: Stage 3's
    cycle-summary presence check (``prompts/stage3.py``) reads only the Mem0
    mirror, never the ledger (see the "Known gap" comment there) — if the
    mirror also went dark whenever the ledger was absent (e.g. a
    deliberately-disabled ``recon_ledger_enabled=False``, a supported
    non-default config), Stage 3 would false-report "summary missing" every
    cycle with no fallback signal. (Reviewer finding robustness, task 2229
    amendment pass.) The return value reflects ONLY the authoritative ledger
    upsert.

    Args:
        memory_service: Service that may expose a ``recon_ledger``
            (:class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore`)
            attribute. Missing/``None`` => the ledger upsert is skipped
            (returns ``False``); the Mem0 mirror and pool-cap trim still run.
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
        remediation: Whether this write is from a focused remediation pass
            rather than a full cycle (task 2652). A remediation pass still
            runs a real Stage 1 (``memory_consolidator``) turn and may emit
            findings — the only thing it skips, by design, is Stage 1's own
            per-cycle summary write; it is not "Stage-2-only". Stamped
            verbatim as ``payload['remediation']`` (so it flows into the
            ledger row's ``payload_json``) and as ``metadata['remediation']``
            on the best-effort Mem0 mirror. Defaults to ``False``, which is
            stamped explicitly (not omitted) — only rows written before this
            change lack the key entirely. Lets
            :meth:`~fused_memory.services.memory_service.MemoryService.get_cycle_summary_presence`
            disambiguate a remediation run's expected missing Stage 1
            (``memory_consolidator``) cycle_summary from a genuine Stage 1
            write failure — see ``prompts/stage3.py``'s Remediation Run
            Exception.

    Returns:
        ``True`` when the authoritative ledger upsert succeeded, ``False``
        otherwise (no ledger wired, or the upsert raised). This reflects
        ONLY the ledger upsert — the Mem0 mirror can succeed even when this
        is ``False`` (e.g. ``recon_ledger_enabled=False``). Callers naming a
        stat off this return value should include "ledger" in the name
        (e.g. ``stageN_cycle_summary_ledger_written``, not a bare
        ``stageN_cycle_summary_written``) so the stat cannot be misread as
        "no summary was written at all" (reviewer finding observability,
        task 2229 amendment pass round 2).
    """
    ledger = getattr(memory_service, 'recon_ledger', None)
    now_dt = _assume_utc(now or datetime.now(UTC))
    written = False
    payload: dict | None = None
    try:
        # Payload construction lives inside this try (not just the upsert
        # call) so a malformed report — e.g. a stray None
        # started_at/completed_at — degrades to written=False + WARNING
        # like any other ledger-write failure, rather than raising an
        # unhandled AttributeError out of this fail-safe function. Built
        # even when `ledger` is None: the best-effort mirror below needs it
        # regardless of ledger availability (see the docstring above).
        payload = {
            'stage': stage,
            'run_id': run_id,
            'started_at': report.started_at.isoformat(),
            'completed_at': report.completed_at.isoformat(),
            'items_flagged_count': len(report.items_flagged),
            'stats': report.stats,
            'llm_calls': report.llm_calls,
            'tokens_used': report.tokens_used,
            'remediation': remediation,
        }
        if ledger is not None:
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
            written = True
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
                # record_type discriminates this deterministic code mirror
                # (LEDGER_STAMP) from the distinct LLM-authored reconstruction
                # write in prompts/stage2.py (NARRATIVE) — task 2468.
                # _apply_cycle_summary_metadata_tagging (memory_service.py)
                # is additive-only and never strips unknown keys, so this
                # survives through to storage unchanged.
                metadata={
                    'kind': 'cycle_summary',
                    'stage': stage,
                    'run_id': run_id,
                    'record_type': CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP,
                    # Mirror-only copy, stamped for parity/observability with
                    # the ledger payload above. No read path consumes it —
                    # get_cycle_summary_presence reads solely from the
                    # ledger's payload_json['remediation'] (the authoritative
                    # source); do not add a fallback read of this mirror copy
                    # (task 2652).
                    'remediation': remediation,
                },
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
