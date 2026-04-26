"""Flag deduplication helpers for Stage 1 (MemoryConsolidator).

This module provides deterministic, code-level deduplication of Stage 1's
``items_flagged`` output.  The LLM has no memory of prior cycles, so the same
(task_id, flag_type) pair can be emitted cycle after cycle.  Rather than
relying on prompt instructions (fragile across model versions), the harness
post-processes each report: for flags with a computable *signature* we check
Mem0 for a prior ``stage1_flag_marker`` memory and, on a hit, annotate the
flag with ``persisted_from_run`` so Stage 2 can decide whether to re-act.

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
    """Deduplicate Stage 1 flagged items against prior ``stage1_flag_marker`` memories.

    For each flag in *flags*:

    - If the flag has no computable signature (missing ``task_id`` or
      ``flag_type``), it is returned unchanged — no I/O performed.
    - If a signature is computable, Mem0 is searched for a prior marker memory
      with matching ``task_id`` and ``flag_type``.  On a hit the flag is
      annotated with ``persisted_from_run`` and ``last_seen_run_id``, and a
      refresh marker is written.  On a miss a new marker is written.
    - All search/write exceptions are caught and logged at WARNING so that a
      transient Mem0 outage does not abort the stage run.

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
            matches = await memory_service.search(
                query=f'stage1 flag marker task {tid} type {ftype}',
                project_id=project_id,
                categories=['observations_and_summaries'],
                limit=10,
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
                prior_run_id = (prior.metadata or {}).get('run_id', run_id)
                flag = dict(flag)
                flag['persisted_from_run'] = prior_run_id
                flag['last_seen_run_id'] = run_id
                # Write a refresh marker so last_seen_run_id is up to date
                await memory_service.add_memory(
                    content=f'Stage 1 flag marker: task={tid} type={ftype} from run={prior_run_id}',
                    category='observations_and_summaries',
                    project_id=project_id,
                    metadata={
                        'source': 'stage1_flag_marker',
                        'task_id': tid,
                        'flag_type': ftype,
                        'run_id': prior_run_id,
                        'last_seen_run_id': run_id,
                    },
                    causation_id=run_id,
                    _source='reconciliation',
                )
        except Exception as e:
            logger.warning('flag_dedup failed for task %s flag_type %s: %s', tid, ftype, e)
        result.append(flag)
    return result


def compute_flag_signature(flag: dict[str, Any]) -> tuple[str, str] | None:
    """Return a (task_id_str, flag_type_str) signature for *flag*, or ``None``.

    Both ``task_id`` and ``flag_type`` must be present and truthy for a
    signature to be computed.  Values are coerced to ``str`` so that an integer
    task_id (common in LLM output) and a string task_id compare equal.

    Returns ``None`` for flags without enough signal to deduplicate — these are
    passed through unchanged by :func:`dedup_flags`.
    """
    task_id = flag.get('task_id')
    flag_type = flag.get('flag_type')
    if not task_id or not flag_type:
        return None
    return (str(task_id), str(flag_type))
