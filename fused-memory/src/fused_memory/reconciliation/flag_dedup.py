"""Flag deduplication helpers for Stage 1 (MemoryConsolidator).

This module provides code-level annotation of Stage 1's ``items_flagged``
output.  The LLM has no memory of prior cycles, so the same (task_id,
flag_type) pair can be emitted cycle after cycle.  For flags with a
computable *signature* we check Mem0 for a prior ``stage1_flag_marker``
memory.  On a hit, the flag is annotated with ``persisted_from_run`` — this
is advisory: Stage 2's prompt uses the annotation to decide whether to
re-act, search for prior actions, or skip the flag.  On a miss, a new
marker memory is written so future cycles can detect the repeat.

Note: this module does **not** suppress persistent flags before Stage 2 sees
them; suppression logic lives in Stage 2's prompt instructions which direct
the LLM to soft-handle annotated flags.

Marker growth
-------------
Markers accumulate monotonically — one row per unique (task_id, flag_type)
pair.  No refresh write is made on subsequent hits (see :func:`dedup_flags`
for rationale).  Stale markers must be purged manually; there is no automated
GC sweep.

Public API
----------
- ``compute_flag_signature(flag)`` — cheap, sync, no I/O.
- ``dedup_flags(memory_service, project_id, run_id, flags)`` — async, does
  Mem0 search + optional write; best-effort (exceptions are logged, not raised).
"""
from __future__ import annotations

import logging
from typing import Any

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
      with matching ``task_id`` and ``flag_type``.  On a hit the flag is
      annotated with ``persisted_from_run`` and ``last_seen_run_id``; no
      additional marker write is made (avoids monotonic marker accumulation).
      On a miss a new marker is written so future cycles detect the repeat.
    - All search/write exceptions are caught and logged at WARNING so that a
      transient Mem0 outage does not abort the stage run.

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
        try:
            # Limit=50 and constrain by category to reduce false-negatives from
            # embedding-similarity ranking when the marker table is large.
            matches = await memory_service.search(
                query=f'stage1 flag marker task {tid} type {ftype}',
                project_id=project_id,
                categories=['observations_and_summaries'],
                limit=50,
            )
            prior = next(
                (
                    r for r in matches
                    if (
                        (r.metadata or {}).get('source') == 'stage1_flag_marker'
                        and str((r.metadata or {}).get('task_id', '')) == tid
                        and str((r.metadata or {}).get('flag_type', '')) == ftype
                    )
                ),
                None,
            )
            if prior is not None:
                prior_run_id = (prior.metadata or {}).get('run_id') or 'unknown'
                flag = dict(flag)
                flag['persisted_from_run'] = prior_run_id
                flag['last_seen_run_id'] = run_id
                # No refresh write — the prior marker is sufficient for dedup.
                # Appending a new marker every cycle would grow the marker table
                # monotonically (N*M rows for M flags over N cycles).
                #
                # Note: last_seen_run_id is set once at marker *creation* and is
                # NOT refreshed on subsequent hits.  A GC sweep keyed on that
                # field would therefore expire markers that are still actively
                # being matched — defeating the dedup purpose.  Stale markers
                # must be purged manually (see module docstring).
            else:
                # Novel flag — write a new marker for future dedup cycles.
                # _source='stage1_flag_dedup' distinguishes these from
                # 'targeted_recon' writes in the audit journal.
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
