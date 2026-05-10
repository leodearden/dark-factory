"""Flag deduplication helpers for Stage 1 (MemoryConsolidator).

This module provides code-level annotation of Stage 1's ``items_flagged``
output.  The LLM has no memory of prior cycles, so the same (task_id,
flag_type) pair can be emitted cycle after cycle.  For flags with a
computable *signature* we check Mem0 for a prior ``stage1_flag_marker``
memory.  On a hit, the flag is annotated with ``persisted_from_run`` and a
replacement marker is written, then all prior markers are deleted (best-effort
replacement).  Two concurrent dedup_flags calls for the same (task_id,
flag_type) may both write replacements and both delete the shared prior,
leaving up to N transient duplicate markers; the next cycle's HIT branch
collapses them back to one.  On a miss, a new marker memory is written so
future cycles can detect the repeat.

Note: this module does **not** suppress persistent flags before Stage 2 sees
them; suppression logic lives in Stage 2's prompt instructions which direct
the LLM to soft-handle annotated flags.

Authoritative suppression gate (task-1186)
------------------------------------------
``dedup_flags`` now calls ``filter_suppressed`` as its **first step**, before
the existing signature-dedup loop.  ``filter_suppressed`` performs one
project-scoped Mem0 search per ``dedup_flags`` call to retrieve all active
``stage1_flag_suppression`` records.  Flags whose ``task_id`` matches a
suppression record are dropped entirely; the remaining flags proceed through
the signature-dedup loop unchanged.  This enforces the suppression contract
in code, making it authoritative over the LLM-side prompt directive.

Best-effort replacement contract (task-1146, hardened in task-1165)
--------------------------------------------------------------------
On every HIT the dedup flow is:

1. Find ALL prior markers for (task_id, flag_type) via ``find_prior_memories``
   (plural); sort by id lex; annotation extracted from the lowest-id-lex
   prior before any deletes.
2. Write a new replacement marker with the current ``run_id``.
3. Only if the write succeeds **and** Mem0 confirmed it (non-empty
   ``memory_ids`` in the response): delete every prior marker (per-prior
   try/except WARNING so one bad delete does not abort the batch).  The
   empty-memory_ids guard prevents a silent Mem0 no-op from wiping priors
   and leaving no dedup state for the next cycle.

This is self-healing: even if past leakage produced N prior markers, the next
dedup_flags call collapses them to a single row.  Write-first ordering with
the empty-memory_ids guard provides best-effort at-least-one-marker: either
the new marker exists (proceed to delete priors) or write failed/was a no-op
(priors intact for next cycle).  Note that write+delete are two separate
non-atomic steps; a crash between them leaves a transient duplicate, which
the next cycle's HIT branch reclaims.

Reclamation bound: ``find_prior_memories`` is called with ``limit=50``, so if
past leakage produced more than 50 markers for one (task_id, flag_type) pair,
each cycle reclaims at most 50 of them.  In practice leakage is bounded by
the number of outage cycles (transient Mem0 failures) and is expected to
remain far below 50; the self-healing property still holds over multiple
cycles.

Public API
----------
- ``compute_flag_signature(flag)`` — cheap, sync, no I/O.
- ``filter_suppressed(memory_service, project_id, flags)`` — async, one
  project-scoped Mem0 search; drops suppressed flags before signature dedup.
- ``dedup_flags(memory_service, project_id, run_id, flags)`` — async, calls
  ``filter_suppressed`` first then does Mem0 search + write + delete per flag;
  best-effort (exceptions are logged, not raised).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.mem0_dedup import find_prior_memories

logger = logging.getLogger(__name__)


class _SuppressionMetadata(TypedDict):
    kind: Literal['stage1_flag_suppression']
    task_id: int


class SuppressionPayload(TypedDict):
    """Canonical Mem0 payload shape for a ``stage1_flag_suppression`` record.

    Enforces the schema documented in the ``## Flag Suppression Check`` section
    of ``STAGE1_SYSTEM_PROMPT`` at the type level so that mis-typed callers are
    caught by mypy rather than silently accepted.
    """

    content: str
    category: Literal['observations_and_summaries']
    metadata: _SuppressionMetadata


async def filter_suppressed(
    memory_service: Any,
    project_id: str,
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop flags whose ``task_id`` has an active ``stage1_flag_suppression`` record.

    Performs one project-scoped Mem0 search per call to retrieve all active
    suppression records.  Suppressed task_ids are extracted into a set; flags
    whose ``task_id`` is in that set are dropped.  The remaining flags are
    returned unchanged so they can proceed through the signature-dedup loop.

    Canonical suppression record schema (owned by the producer task):
    - ``metadata.kind == "stage1_flag_suppression"``
    - ``metadata.task_id == <N>`` (int or str; coerced to str on both sides)

    Both fields must be present and correct for a record to be treated as a
    suppression — this rejects vector-search near-misses that only match on
    semantic proximity.  ``task_id`` values that are ``None`` or ``''`` in a
    suppression record are skipped (invalid; not added to the suppressed set),
    preventing a malformed record from accidentally suppressing flags that have
    no task_id.

    The search uses ``limit=500``.  Projects with more than 500 active
    suppression records would see the excess truncated; in practice suppression
    records are a small operator-managed set and 500 provides ample headroom.

    On search exception: logs a WARNING and returns *flags* unchanged
    (conservative pass-through — treats as "no suppression in effect").
    """
    if not flags:
        return []

    try:
        results = await memory_service.search(
            query='stage1_flag_suppression',
            project_id=project_id,
            categories=['observations_and_summaries'],
            stores=['mem0'],
            limit=500,
        )
    except Exception as e:
        logger.warning(
            'filter_suppressed search failed for project %s: %s', project_id, e
        )
        return flags

    suppressed_task_ids: set[str] = set()
    for r in results:
        meta = r.metadata or {}
        if meta.get('kind') != 'stage1_flag_suppression':
            continue
        task_id = meta.get('task_id')
        if task_id is None or task_id == '':
            continue
        suppressed_task_ids.add(str(task_id))

    def _keep(f: dict[str, Any]) -> bool:
        flag_tid = f.get('task_id')
        if flag_tid is None or flag_tid == '':
            return True  # symmetric with producer guard at lines 149-150
        return str(flag_tid) not in suppressed_task_ids

    return [f for f in flags if _keep(f)]


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
        succeeds and Mem0 confirms it, delete the prior marker
        (best-effort replacement pattern).
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
    # --- Authoritative suppression gate (task-1186) ---
    # Drop flags for tasks with active stage1_flag_suppression records BEFORE
    # the signature-dedup loop so suppressed flags never reach the per-flag
    # prior-marker write path.
    flags = await filter_suppressed(memory_service, project_id, flags)

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
            # --- HIT: best-effort replacement ---
            # (1) Sort priors by id lex so annotation source and deletion order
            #     are deterministic across concurrent cycles.  MemoryResult.id
            #     is always present (str); temporal fields are optional and may
            #     be absent for Mem0 results, so lex sort is the only total order.
            priors = sorted(priors, key=lambda p: p.id)
            # Extract annotation from the first prior (lowest id lex) BEFORE
            # deleting any.
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
            #     Also check that Mem0 actually persisted the write (non-empty
            #     memory_ids); a silent no-op (empty memory_ids) skips deletion
            #     so priors are preserved for the next cycle.
            write_succeeded = False
            try:
                response = await memory_service.add_memory(
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
                if getattr(response, 'memory_ids', None):
                    write_succeeded = True
                else:
                    logger.warning(
                        'flag_dedup: add_memory returned no memory_ids on HIT for task %s'
                        ' flag_type %s — skipping prior deletion',
                        tid, ftype,
                    )
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
            # Marker-growth caveat: when find_prior_memories returns [] due to
            # a search failure (transient Mem0 outage) rather than a genuine
            # miss, this branch still writes a new marker.  During a sustained
            # outage every cycle will write a marker for recurring flags,
            # causing monotonic marker-table growth beyond the normal one-row-
            # per-(task_id, flag_type) bound.  The best-effort replacement
            # pattern on the HIT path ensures that once search recovers, the
            # next cycle collapses any accumulated duplicates back to a single row.
            try:
                miss_response = await memory_service.add_memory(
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
                if not getattr(miss_response, 'memory_ids', None):
                    logger.warning(
                        'flag_dedup: add_memory returned no memory_ids on MISS for task %s'
                        ' flag_type %s — recurring flag will not be detected next cycle',
                        tid, ftype,
                    )
            except Exception as e:
                logger.warning('flag_dedup failed for task %s flag_type %s: %s', tid, ftype, e)
        result.append(flag)
    return result


def build_suppression_payload(task_id: int | str) -> SuppressionPayload:
    """Build the canonical ``stage1_flag_suppression`` Mem0 payload for *task_id*.

    Returns a :class:`SuppressionPayload` with ``content``, ``category``, and
    ``metadata`` fields matching the canonical schema documented in the
    ``## Flag Suppression Check`` section of ``STAGE1_SYSTEM_PROMPT``.
    ``task_id`` is coerced to ``int`` so the producer always pins the integer
    type regardless of how the caller obtained the id.

    ``project_id`` is intentionally absent — it is a write-time concern that
    must be passed separately to ``memory_service.add_memory``, keeping this
    helper pure and reusable across projects.

    Canonical schema (Mem0, observations_and_summaries category):
      - ``metadata.kind = "stage1_flag_suppression"``
      - ``metadata.task_id = <N>`` (int — coerced by this function)
      - ``content = "STAGE 1 FLAG SUPPRESSION task_id=<N>"``
    """
    tid = int(task_id)
    return {
        'content': f'STAGE 1 FLAG SUPPRESSION task_id={tid}',
        'category': 'observations_and_summaries',
        'metadata': {
            'kind': 'stage1_flag_suppression',
            'task_id': tid,
        },
    }


async def write_suppression_record(
    memory_service: Any,
    *,
    project_id: str,
    task_id: int | str,
    causation_id: str | None = None,
) -> AddMemoryResponse:
    """Write a ``stage1_flag_suppression`` record to Mem0 for *task_id*.

    Builds the canonical payload via :func:`build_suppression_payload` (which
    coerces *task_id* to ``int`` and pins ``metadata.kind``/``content``) then
    calls ``memory_service.add_memory`` with *project_id* and *causation_id*
    as separate write-time kwargs.

    The ``_source='stage1_flag_suppression'`` sentinel distinguishes these
    writes from ``'stage1_flag_dedup'`` and ``'targeted_recon'`` writes in the
    audit journal, enabling per-class retention and query filtering.

    Canonical schema (Mem0, observations_and_summaries category):
      - ``metadata.kind = "stage1_flag_suppression"``
      - ``metadata.task_id = <N>`` (int — coerced by build_suppression_payload)
      - ``content = "STAGE 1 FLAG SUPPRESSION task_id=<N>"``

    Returns the :class:`AddMemoryResponse` from the memory service so callers
    can inspect ``memory_ids`` for empty-list deduplication / no-op detection.
    """
    payload = build_suppression_payload(task_id)
    return await memory_service.add_memory(
        **payload,
        project_id=project_id,
        causation_id=causation_id,
        _source='stage1_flag_suppression',
    )


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
