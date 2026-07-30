"""Recon-initiated Mem0 deletion tombstones and the protected-mirror predicate
(task 3041).

Two independent units live here, both consumed by every recon path that
deletes a Mem0 record — the marker-GC sweeps in
``reconciliation.stages.task_knowledge_sync._sweep_stale_mem0_pool`` and the
cycle_summary mirror pool trim in
``reconciliation.summary_pool.enforce_summary_pool_cap``.

Why this module exists
----------------------
Recon gate 165 (esc-165-1) reported three cycle_summary Mem0 anchors for run
``84eae9bd`` as silently lost. They were not lost: they were evicted, exactly
as designed, by the cap-2 mirror pool trim inside ``write_cycle_summary``
(the Mem0 mirror is documented as "a best-effort searchable copy of the
ledger's authoritative payload, not itself a source of truth" — the
authoritative rows survived and are still readable via
``get_cycle_summary_presence``). The DEFECT was that a designed eviction was
indistinguishable from fleet-wide data loss: ``get_memory_by_id`` answered
``{found: false}`` and there was no reachable path from a memory uuid to
"who deleted it and why". :func:`record_mem0_deletion_tombstone` closes that
gap; :func:`is_protected_mirror_record` closes the adjacent precision gap
that made an over-broad marker sweep *able* to take a mirror at all.

The two discriminator vocabularies
----------------------------------
:func:`is_protected_mirror_record` checks BOTH ``kind`` and ``record_type``
because they are independent vocabularies answering different questions:

- ``kind == 'cycle_summary'`` is the POOL IDENTITY. It is what
  ``services.memory_service._apply_cycle_summary_metadata_tagging`` keys on
  to auto-stamp ``recon_pool``, and it is shared by both writers of a
  cycle_summary record.
- ``record_type == 'ledger_stamp'`` (task 2468) DISCRIMINATES the
  deterministic Python-written mirror of the authoritative ledger row from
  the LLM-authored ``'narrative'`` reconstruction write in
  ``reconciliation.prompts.stage2``. Task 2468 shipped ``record_type`` as
  deliberately write-only ("no reader ... one lands alongside the tooling
  that needs it"); this module is its first real reader.

Checking both means a record missing either tag — an LLM narrative that
omits ``record_type``, or a mirror whose ``kind`` was overwritten — is still
protected. The predicate is deliberately OR, not AND: over-protecting a
marker costs one skipped GC (loudly logged), while under-protecting a mirror
costs the audit anchor an auditor is about to look for.

Why the predicate, not a tighter payload filter
-----------------------------------------------
This predicate exists so a marker-GC sweep can never take a cycle_summary
mirror REGARDLESS of how loose its Qdrant payload filter is. That is not
hypothetical: ``task_knowledge_sync._FLAG_FOR_STAGE2_ENUM_FILTERS`` is
``{'flag_for_stage2': True}`` with no ``kind``/``source``/``record_type``
discriminator at all, and ``flag_for_stage2`` is an LLM-supplied metadata
key that nothing at the ``add_memory`` boundary stops a cycle_summary write
from also carrying. No amount of payload-filter tightening can guarantee
precision when the matched key is one any writer can attach to any record —
so the guard is enforced at the shared ``_sweep_stale_mem0_pool`` choke
point instead, where every current and future sweep inherits it.

Import posture
--------------
NEAR-LEAF: imports only ``reconciliation.recon_ledger`` — the same single
import edge ``reconciliation.summary_pool`` already has. It must NOT import
``summary_pool`` (which imports this module for the trim-path tombstone
write, so that edge would be a cycle) nor anything under
``fused_memory.services`` (see ``recon_pool_map.py``'s docstring for the
memory_consolidator -> task_knowledge_sync -> services -> memory_service
cycle this package is careful to avoid).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord

logger = logging.getLogger(__name__)

# ReconLedgerStore record_kind for a recon-initiated Mem0 deletion tombstone.
#
# Deliberately NOT a member of recon_ledger.MARKER_KINDS: that tuple drives
# gc()'s terminal-task DELETE arm and marker_task_ids(), both of which treat
# the task_id column as a Taskmaster task id. A tombstone's task_id column
# holds a Mem0 memory uuid, so it must never enter either path — it expires
# purely via expires_at.
RECORD_KIND_MEM0_TOMBSTONE: str = 'mem0_tombstone'

# Retention window for tombstone rows. Follows the same rationale already
# written down for summary_pool.CYCLE_SUMMARY_TTL_DAYS: the recon ledger is a
# bounded control-plane store, not a permanent audit log, so rows get a TTL
# and are reaped by the existing ReconLedgerStore.gc() expires_at pass (run
# every cycle by stages.task_knowledge_sync._gc_recon_markers) rather than
# kept forever or given bespoke cleanup code. 30 days comfortably outlives
# the window in which an auditor investigates a missing record.
MEM0_TOMBSTONE_TTL_DAYS: int = 30

# Kept in sync BY CONVENTION with summary_pool.CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP
# (task 2468) rather than imported from it: summary_pool imports THIS module
# (enforce_summary_pool_cap writes a tombstone per trimmed member), so an
# import in the other direction would be a cycle. If either literal is ever
# edited, edit both — summary_pool.py carries the reciprocal note.
_RECORD_TYPE_LEDGER_STAMP: str = 'ledger_stamp'

# metadata.kind identifying the cycle_summary mirror pool. Mirrored from the
# literal written by summary_pool.write_cycle_summary's add_system_record
# call and by services.memory_service._apply_cycle_summary_metadata_tagging;
# same no-cycle rationale as above.
_KIND_CYCLE_SUMMARY: str = 'cycle_summary'

# Identifying victim keys copied into a tombstone payload. Deliberately an
# identity-only projection — never the record's content — so a tombstone
# stays small and duplicates no payload it is merely accounting for.
_VICTIM_IDENTITY_KEYS: tuple[str, ...] = (
    'kind',
    'record_type',
    'source',
    'recon_pool',
    'run_id',
)


def _assume_utc(dt: datetime) -> datetime:
    """Return *dt* with UTC attached if naive; return *dt* unchanged otherwise.

    Same "naive datetimes are UTC" convention as
    ``reconciliation.summary_pool._assume_utc`` and
    ``reconciliation.stages.task_knowledge_sync._assume_utc``; duplicated for
    the same reason those two are (this module must stay near-leaf and cannot
    import either of them).
    """
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def is_protected_mirror_record(metadata) -> bool:
    """Return True when *metadata* identifies a cycle_summary ledger mirror.

    A record matching this predicate must never be deleted by a marker-GC
    sweep. See the module docstring for why both discriminators are checked
    and why the guard lives here rather than in each sweep's payload filter.

    Fully defensive: ``None``, a non-dict, or a dict whose values are of an
    unexpected type all return ``False`` without raising. A marker sweep runs
    against whatever Mem0 hands back and must never crash on a weird payload.

    Args:
        metadata: A Mem0 record's ``metadata`` mapping, or anything at all.

    Returns:
        ``True`` when the payload declares ``kind == 'cycle_summary'`` OR
        ``record_type == 'ledger_stamp'``; ``False`` otherwise.
    """
    if not isinstance(metadata, dict):
        return False
    return (
        metadata.get('kind') == _KIND_CYCLE_SUMMARY
        or metadata.get('record_type') == _RECORD_TYPE_LEDGER_STAMP
    )


async def record_mem0_deletion_tombstone(
    memory_service,
    project_id: str,
    memory_id: str,
    *,
    victim_metadata: dict | None,
    victim_created_at: str | None,
    deleter: str,
    deleting_run_id: str,
    now: datetime | None = None,
) -> bool:
    """Record that a recon sweep deleted the Mem0 record *memory_id*.

    Writes ONE :class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerRecord`
    with ``record_kind='mem0_tombstone'`` and ``task_id=memory_id``, leaving
    ``flag_type``/``run_id`` at their ``''`` defaults — so a later lookup needs
    only the one thing an auditor actually knows (the memory uuid) to satisfy
    the store's existing five-part ``get_by_identity``. The primary key's
    ``ON CONFLICT`` makes a repeat write for the same victim idempotent (row
    count stays 1), and ``expires_at`` hands retention to the ``gc()`` pass
    ``_gc_recon_markers`` already runs every cycle, so tombstones cannot grow
    unbounded without any new cleanup code.

    **Ordering contract — call this only AFTER the delete is confirmed
    successful.** Both delete paths (``_sweep_stale_mem0_pool`` and
    ``enforce_summary_pool_cap``) already degrade an individual delete failure
    to a WARNING and exclude it from their success count; writing the tombstone
    from the success branch only means a tombstone always means "really gone".
    A pre-delete write would instead false-positive on every failed delete,
    producing an audit trail that claims records are gone while they are alive
    — strictly worse than the status quo this is fixing. The residual window
    (process dies between a successful delete and this write) is ACCEPTED and
    documented: it is still covered by the pre-existing
    ``WriteJournal.log_write_op`` row that ``delete_memory`` emits.

    Fail-safe throughout, because this runs inside a delete sweep: the whole
    body is wrapped so a missing/None ``recon_ledger`` or a raising ``upsert``
    degrades to one WARNING and a ``False`` return. It can never raise into a
    sweep and never alters a sweep's returned count.

    Args:
        memory_service: Service that may expose a ``recon_ledger``
            (:class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore`)
            attribute. Missing/``None`` => no tombstone, returns ``False``.
            Same ``getattr`` optional-store precedent as
            ``summary_pool.write_cycle_summary`` and
            ``stages.task_knowledge_sync._gc_recon_markers``, so a deployment
            running ``recon_ledger_enabled=False`` degrades identically.
        project_id: Project scope for the tombstone row.
        memory_id: The deleted Mem0 record's uuid — stored as ``task_id``.
        victim_metadata: The deleted record's ``metadata`` mapping, or
            ``None``. Only :data:`_VICTIM_IDENTITY_KEYS` are copied — never the
            record's content — so the tombstone stays an accounting row rather
            than a second copy of the payload.
        victim_created_at: The deleted record's ``created_at``, or ``None``.
            Carried verbatim so an auditor can tell how old the evicted record
            was without reconstructing the pool's history.
        deleter: The delete's ``_source`` audit tag, i.e. WHICH sweep took it
            (e.g. ``'stage1_cycle_summary_trim'``,
            ``'stage1_flag_marker_gc_sweep'``).
        deleting_run_id: The run that performed the deletion. Deliberately
            distinct from the victim's own ``metadata['run_id']`` (also stored,
            via ``victim_metadata``) — conflating the two is precisely what
            made the original recon-gate-165 finding unreadable.
        now: Reference "current time". Defaults to ``datetime.now(UTC)``;
            tests inject a fixed value. Normalized via :func:`_assume_utc` and
            rendered with ``.isoformat()`` so ``expires_at`` stays correct
            under the ledger's lexicographic TEXT ``gc()`` comparison.

    Returns:
        ``True`` when the tombstone row was written, ``False`` otherwise (no
        ledger wired, or the upsert failed).
    """
    try:
        ledger = getattr(memory_service, 'recon_ledger', None)
        if ledger is None:
            return False

        now_dt = _assume_utc(now or datetime.now(UTC))
        metadata = victim_metadata if isinstance(victim_metadata, dict) else {}
        payload = {
            'deleter': deleter,
            'deleting_run_id': deleting_run_id,
            'deleted_at': now_dt.isoformat(),
            'created_at': victim_created_at,
            **{key: metadata.get(key) for key in _VICTIM_IDENTITY_KEYS},
        }
        await ledger.upsert(
            ReconLedgerRecord(
                project_id=project_id,
                record_kind=RECORD_KIND_MEM0_TOMBSTONE,
                task_id=memory_id,
                flag_type='',
                run_id='',
                payload_json=json.dumps(payload, default=str),
                state='deleted',
                created_at=now_dt.isoformat(),
                expires_at=(
                    now_dt + timedelta(days=MEM0_TOMBSTONE_TTL_DAYS)
                ).isoformat(),
            )
        )
        return True
    except Exception:
        logger.warning(
            'reconciliation.record_mem0_deletion_tombstone: '
            'tombstone write failed for memory_id=%s deleter=%s; '
            'the delete itself still succeeded and remains journaled',
            memory_id,
            deleter,
            exc_info=True,
            extra={
                'project_id': project_id,
                'memory_id': memory_id,
                'deleter': deleter,
                'run_id': deleting_run_id,
            },
        )
        return False
