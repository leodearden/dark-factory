"""Flag deduplication helpers for Stage 1 (MemoryConsolidator).

This module provides code-level annotation of Stage 1's ``items_flagged``
output.  The LLM has no memory of prior cycles, so the same (task_id,
flag_type) pair can be emitted cycle after cycle.  For flags with a
computable *signature* we check Mem0 for a prior ``stage1_flag_marker``
memory.  On a hit, the flag is annotated with ``persisted_from_run`` and a
replacement marker is written, then all prior markers are deleted (atomic
replacement).  On a miss, a new marker memory is written so future cycles
can detect the repeat.

Note: this module does **not** suppress persistent flags before Stage 2 sees
them; suppression logic lives in Stage 2's prompt instructions which direct
the LLM to soft-handle annotated flags.

Atomic-replacement contract (task-1146)
---------------------------------------
On every HIT the dedup flow is:

1. Find ALL prior markers for (task_id, flag_type) via ``find_prior_memories``
   (plural); annotation extracted from the first result before any deletes.
2. Write a new replacement marker with the current ``run_id``.
3. Only if the write succeeds: delete every prior marker (per-prior try/except
   WARNING so one bad delete does not abort the batch).

This is self-healing: even if past leakage produced N prior markers, the next
dedup_flags call collapses them to a single row.  Write-first ordering
guarantees at-least-one-marker: either the new marker exists (proceed to
delete priors) or write failed (priors intact for next cycle).

Public API
----------
- ``compute_flag_signature(flag)`` — cheap, sync, no I/O.
- ``dedup_flags(memory_service, project_id, run_id, flags)`` — async, does
  Mem0 search + write + delete; best-effort (exceptions are logged, not raised).
"""
from __future__ import annotations

import logging
from typing import Any

from fused_memory.reconciliation.mem0_dedup import find_prior_memories

logger = logging.getLogger(__name__)


async def dedup_flags(
    memory_service: Any,
    project_id: str,
    run_id: str,
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate Stage 1 flagged items against prior ``stage1_flag_marker`` memories.

    For each flag in *flags*:

    - If the flag has no computable signature (missing ``task_id`` or
      ``flag_type``), it is returned unchanged — no I/O performed.
    - If a signature is computable, Mem0 is searched for a prior marker memory
      with matching ``task_id`` and ``flag_type``.
      - On a HIT: annotate the flag with ``persisted_from_run`` and
        ``last_seen_run_id``; write a new replacement marker; if the write
        succeeds, delete the prior marker (atomic-replacement pattern).
      - On a MISS: write a new marker so future cycles detect the repeat.
    - All search/write/delete exceptions are caught and logged at WARNING so
      that a transient Mem0 outage does not abort the stage run.

    ``persisted_from_run`` is set to the ``run_id`` stored in the prior
    marker's metadata.  If that metadata field is absent, ``None``, or an
    empty string (i.e. any falsy value), the literal sentinel ``'unknown'``
    is used instead.  Downstream consumers (Stage 2 prompt, observability
    dashboards) can grep for ``'unknown'`` to detect malformed markers.

    Returns the (possibly annotated) flag list.
    """
    result: list[dict[str, Any]] = []
    for flag in flags:
        sig = compute_flag_signature(flag)
        if sig is None:
            result.append(flag)
            continue
        tid, ftype = sig
        # Delegate search+filter to the shared helper.  find_prior_memories logs a
        # WARNING under logger on search failure and returns [] so the else
        # branch below writes a fresh marker (best-effort on transient Mem0 outage).
        priors = await find_prior_memories(
            memory_service,
            project_id=project_id,
            task_id=tid,
            kind={'source': 'stage1_flag_marker', 'flag_type': ftype},
            query=f'stage1 flag marker task {tid} type {ftype}',
            categories=['observations_and_summaries'],
            limit=50,
            log=logger,
        )
        if priors:
            # --- HIT: atomic-replacement ---
            # (1) Extract annotation from the FIRST prior BEFORE deleting any.
            #     Annotation pinned to first-found prior (earliest known run_id).
            first_prior = priors[0]
            prior_run_id = (first_prior.metadata or {}).get('run_id') or 'unknown'
            if prior_run_id == 'unknown':
                logger.debug(
                    'flag_dedup: prior marker for task=%s flag_type=%s has malformed run_id metadata',
                    tid,
                    ftype,
                )
            flag = dict(flag)
            flag['persisted_from_run'] = prior_run_id
            flag['last_seen_run_id'] = run_id

            # (2) Write replacement marker first.  If this fails, skip the
            #     delete so all priors remain intact for next cycle.
            write_succeeded = False
            try:
                await memory_service.add_memory(
                    content=f'Stage 1 flag marker: task={tid} type={ftype} from run={run_id}',
                    category='observations_and_summaries',
                    project_id=project_id,
                    metadata={
                        'source': 'stage1_flag_marker',
                        'task_id': tid,
                        'flag_type': ftype,
                        'run_id': run_id,
                        'last_seen_run_id': run_id,
                    },
                    causation_id=run_id,
                    _source='stage1_flag_dedup',
                )
                write_succeeded = True
            except Exception as e:
                logger.warning(
                    'flag_dedup: failed to write replacement marker for task %s flag_type %s: %s',
                    tid, ftype, e,
                )

            # (3) Delete ALL priors only if the new marker was successfully written.
            #     Each delete is wrapped individually so one bad delete does not
            #     abort the batch (self-healing: leftovers are retried next cycle).
            if write_succeeded:
                for prior in priors:
                    try:
                        await memory_service.delete_memory(
                            memory_id=prior.id,
                            store='mem0',
                            project_id=project_id,
                            causation_id=run_id,
                            _source='stage1_flag_dedup',
                        )
                    except Exception as e:
                        logger.warning(
                            'flag_dedup: failed to delete prior marker %s for task %s flag_type %s: %s',
                            prior.id, tid, ftype, e,
                        )
        else:
            # MISS: novel flag (or search failed) — write a new marker for future
            # dedup cycles.  _source='stage1_flag_dedup' distinguishes these
            # from 'targeted_recon' writes in the audit journal.
            #
            # Marker-growth caveat: when find_prior_memory returns None due to
            # a search failure (transient Mem0 outage) rather than a genuine
            # miss, this branch still writes a new marker.  During a sustained
            # outage every cycle will write a marker for recurring flags,
            # causing monotonic marker-table growth beyond the normal one-row-
            # per-(task_id, flag_type) bound.  The atomic-replacement pattern
            # on the HIT path ensures that once search recovers, the next cycle
            # collapses any accumulated duplicates back to a single row.
            try:
                await memory_service.add_memory(
                    content=f'Stage 1 flag marker: task={tid} type={ftype} from run={run_id}',
                    category='observations_and_summaries',
                    project_id=project_id,
                    metadata={
                        'source': 'stage1_flag_marker',
                        'task_id': tid,
                        'flag_type': ftype,
                        'run_id': run_id,
                        'last_seen_run_id': run_id,
                    },
                    causation_id=run_id,
                    _source='stage1_flag_dedup',
                )
            except Exception as e:
                logger.warning('flag_dedup failed for task %s flag_type %s: %s', tid, ftype, e)
        result.append(flag)
    return result


def compute_flag_signature(flag: dict[str, Any]) -> tuple[str, str] | None:
    """Return a (task_id_str, flag_type_str) signature for *flag*, or ``None``.

    Both ``task_id`` and ``flag_type`` must be present (i.e. not ``None``) for
    a signature to be computed.  Values are coerced to ``str`` so that an
    integer task_id (common in LLM output) and a string task_id compare equal.
    Falsy-but-valid values like ``task_id=0`` or ``flag_type=''`` are accepted
    — only ``None`` (absent key) triggers a ``None`` return.

    Returns ``None`` for flags without enough signal to deduplicate — these are
    passed through unchanged by :func:`dedup_flags`.
    """
    task_id = flag.get('task_id')
    flag_type = flag.get('flag_type')
    if task_id is None or flag_type is None:
        return None
    return (str(task_id), str(flag_type))
