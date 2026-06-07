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

Post-write confirmation (task-1400, corrected in task-1400 step-15)
--------------------------------------------------------------------
``add_memory`` returns an id in ``memory_ids``, but Mem0 may store the content
under a DIFFERENT canonical id.  ``confirm_marker_persisted`` performs a
read-back search immediately after each write to verify the marker is
*findable* (not just written).  It returns ``True`` iff at least one matching
marker is findable; ``False`` after one retry-miss; never raises.  On a miss
it logs a WARNING and retries the search exactly once; returns ``False``
after a failed retry.  The HIT-branch prior-deletion gate and the MISS-branch
no-op WARNING are both driven off this bool.

Confirmation kind filter is intentionally ASYMMETRIC with the pre-write dedup
search (design decision #6): it additionally includes ``run_id`` (the current
run's id) so that surviving priors from earlier runs cannot masquerade as
confirmation of the current write.  The two searches have different jobs:
the pre-write dedup search asks "does ANY prior exist across all runs?"
(must be run_id-agnostic); the confirmation search asks "did MY write for
THIS run land?" (must be run_id-scoped).  Scoping confirmation by run_id is
correct on both paths — HIT (new marker run_id=current matches; priors'
older run_ids do not) and MISS (new marker still matches).

Confirmation circuit-breaker (task-1412)
-----------------------------------------
During a sustained Mem0 brownout every flag in the batch incurs the worst-case
confirmation cost (initial search miss + retry = 2 search calls per flag),
compounding pressure on the already-failing backend.  To limit this, ``dedup_flags``
maintains a **per-invocation** circuit-breaker:

- ``consecutive_confirmation_misses`` counts strictly-consecutive ``confirm_marker_persisted``
  misses (``False`` return).  The counter resets to 0 on any successful confirmation
  (``True`` return) so sporadic misses during otherwise-healthy operation do **not**
  accumulate toward the threshold.
- ``confirmation_disabled`` starts ``False`` and is set ``True`` the first time
  ``consecutive_confirmation_misses >= _CONFIRMATION_MISS_THRESHOLD``.
- At the moment of trip, exactly **one** breaker WARNING is logged (format:
  ``"flag_dedup: confirmation circuit-breaker tripped after N consecutive misses;
  falling back to memory_ids gate for remainder of batch"``).  Subsequent flags
  do **not** re-emit the WARNING even if they also miss.
- Once tripped, both HIT and MISS branches skip ``confirm_marker_persisted``
  entirely and fall back to the cheaper pre-task-1400 gate:

  * **HIT branch**: ``write_succeeded = bool(response.memory_ids)``.  Deletion
    of prior markers proceeds if ``True``; is skipped (with a per-flag
    "skipping prior deletion" WARNING) if ``False``.
  * **MISS branch**: the "will not be detected next cycle" WARNING fires only
    if ``bool(miss_response.memory_ids) is False``; silent if ``True``.

- Both branches share **one counter** (same local variable), so a HIT-branch
  trip persists into MISS-branch flags later in the same batch and vice versa.
- The counter and disabled flag are **function-local**, so a subsequent
  ``dedup_flags`` invocation (next reconciliation cycle) gets a fresh budget.
  This is intentional — a transient brownout should not permanently disable
  confirmation for future healthy cycles.
- **Write failures** (``add_memory`` exceptions) do **NOT** count toward the
  threshold.  The circuit-breaker targets specifically the *confirmation* cost
  — the extra search round-trip that ``confirm_marker_persisted`` performs
  after a successful write.  When ``add_memory`` itself raises, the
  confirmation call is never reached, so neither the counter nor the disabled
  flag is touched; the except branch logs a WARNING and moves on.  A sustained
  brownout that manifests as write failures therefore never trips the breaker —
  this is intentional because there is no confirmation overhead to shed when
  writes are failing outright.

The breaker is an **internal load-shedding mechanism** operating entirely within
``dedup_flags``.  It does not change the contract documented in the LLM-side
``stage1.py`` prompt (which mirrors the confirmation contract under normal
conditions).  No change to ``confirm_marker_persisted`` itself is required —
only the call-site within ``dedup_flags`` is gated.

WARNING wording disambiguation (task-1413): the per-flag WARNINGs use distinct
templates for the ACTIVE-breaker miss path (a confirmation search was attempted
and returned no result) versus the TRIPPED-breaker skip path (no search
attempted because the breaker is open).  The ACTIVE wording is
``'could not be confirmed findable'``; the TRIPPED wording is
``'confirmation skipped (circuit-breaker open) and memory_ids gate failed'``.
During a brownout this lets operators distinguish genuine confirmation misses
from gate-only flags raised purely from the memory_ids check.

Public API
----------
- ``compute_flag_signature(flag)`` — cheap, sync, no I/O.
- ``confirm_marker_persisted(memory_service, *, project_id, task_id, flag_type, run_id, log)``
  — async, post-write confirmation search; returns True if findable, False otherwise.
- ``filter_suppressed(memory_service, project_id, flags)`` — async, one
  project-scoped Mem0 search; drops suppressed flags before signature dedup.
- ``dedup_flags(memory_service, project_id, run_id, flags)`` — async, calls
  ``filter_suppressed`` first then does Mem0 search + write + confirm + delete
  per flag; best-effort (exceptions are logged, not raised).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Literal, TypedDict

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.mem0_dedup import find_prior_memories

logger = logging.getLogger(__name__)

# Module-local sleep binding — allows tests to patch sleep without touching the
# global asyncio namespace (same pattern as harness.py).
_sleep = asyncio.sleep


class _SuppressionMetadata(TypedDict):
    """Producer-side contract: ``task_id`` is pinned to ``int``.

    Reader (``filter_suppressed``) tolerates and str-coerces both ``int``
    and ``str`` task_ids for backward compat with legacy hand-authored
    records — do NOT tighten the reader to int-only without a migration
    of any pre-existing str-task_id records in Mem0.
    """

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


# Number of *consecutive* confirm_marker_persisted misses within one dedup_flags
# call that triggers the confirmation circuit-breaker.  Once tripped the
# remaining flags in the batch skip confirm_marker_persisted and fall back to
# bool(response.memory_ids) as the write-succeeded gate.  Counter is
# function-local so each dedup_flags invocation starts with a fresh budget.
# Tests monkeypatch this down to 2 (same idiom as test_durable_queue.py:635
# with _DELETE_DEAD_BATCH_SIZE).  See "Confirmation circuit-breaker (task-1412)"
# section in the module docstring for the full design rationale.
#
# Trade-off: resilience to single-flag flakiness (higher threshold) vs
# brownout load-shedding latency (lower threshold).
#
# 3 was chosen as the default (task-1415, lowered from 5):
# - confirm_marker_persisted already retries internally so each "miss" costs 2
#   search round-trips.  At threshold 5 the worst-case batch pays up to
#   5 × (1 write + 2 confirmation searches) ≈ 15 round-trips before the breaker
#   activates; at 3 that drops to ≈ 9.
# - Threshold 3 still tolerates a single spurious miss without tripping: a
#   sporadic miss followed by a hit resets the counter to 0, so strictly-
#   consecutive miss runs of 3 are rare under healthy-but-slow indexing.
# - The whole point of the breaker is to shed load during real brownouts, so
#   activating it sooner (lower threshold) is consistent with its purpose.
# - Write failures (add_memory exceptions) do NOT count — the counter only
#   advances on confirmation misses from successful writes; see the module
#   docstring "Confirmation circuit-breaker" section for the "write failures do
#   not count" design rationale.
_CONFIRMATION_MISS_THRESHOLD: int = 3

# Bounded delay (seconds) awaited between the first-search miss and the retry in
# confirm_marker_persisted.  Default 0.0 = pure event-loop yield (asyncio.sleep(0)
# semantics): yields control to the loop, costs nothing on happy paths, and preserves
# the module docstring's "Mem0 writes assumed to be immediately visible" invariant.
# Bump via monkeypatch in tests or via future config if production shows a Mem0
# write-flush boundary — this is the knob the docstring "Mem0 read-after-write
# consistency" paragraph anticipates ("If production evidence shows otherwise, add a
# small bounded delay before the retry").
_CONFIRM_RETRY_DELAY_SECS: float = 0.0


def _marker_query(tid: str, ftype: str) -> str:
    """Build the canonical Mem0 search query for a stage1_flag_marker.

    Single source of truth used by BOTH:
    - The pre-write dedup search in ``dedup_flags`` (asks "does any prior exist
      across all runs?"; run_id-agnostic).
    - The post-write confirmation search in ``confirm_marker_persisted``
      (asks "did MY write for THIS run land?"; run_id-scoped via a separate
      ``kind`` filter — see that helper for details).

    The two callers differ in their ``kind`` filter but use the SAME query
    string.  Tests rely on this equality to dispatch a single marker-search
    stub from two call sites.
    """
    return f'stage1 flag marker task {tid} type {ftype}'


async def confirm_marker_persisted(
    memory_service: Any,
    *,
    project_id: str,
    task_id: str,
    flag_type: str,
    run_id: str,
    log: logging.Logger,
) -> bool:
    """Confirm a just-written ``stage1_flag_marker`` is findable by a subsequent search.

    Performs a read-back search to confirm findability; returns ``True`` iff at
    least one matching marker is returned (initial search or retry), ``False``
    otherwise.

    The confirmation kind filter includes ``run_id`` (the current run) so that
    surviving priors from earlier runs cannot masquerade as confirmation of this
    write.  This is intentionally ASYMMETRIC with the pre-write dedup search
    (which omits ``run_id`` so it can find priors from any earlier run).
    The two searches have different jobs:

    - Pre-write dedup: "does ANY prior exist across all runs?" → run_id-agnostic.
    - Confirmation:    "did MY write for THIS run land?"       → run_id-scoped.

    Strategy:
    1. Run a confirmation search via ``find_prior_memories`` with
       ``kind={'source':'stage1_flag_marker','flag_type':flag_type,'run_id':run_id}``.
    2. If matches are found, return ``True``.
    3. On a miss, log a WARNING (task_id + flag_type), await
       ``_sleep(_CONFIRM_RETRY_DELAY_SECS)`` (default 0.0 = pure event-loop yield),
       then retry the search once.
    4. Return ``True`` if the retry finds matches; otherwise log a final WARNING
       and return ``False``.
    5. Never raises — the whole body is wrapped in a best-effort try/except so
       a non-search error path cannot abort ``dedup_flags``.

    Mem0 read-after-write consistency:
        Flag markers use ``category='observations_and_summaries'`` which routes
        to Mem0 (not Graphiti).  The indexing-lag caveat in ``prompts/stage1.py``
        (lines 189-196) is specific to Graphiti's async embedding pipeline and
        does NOT apply here — Mem0 writes on this path are assumed to be
        immediately visible to a subsequent ``search``.  The configurable
        ``_CONFIRM_RETRY_DELAY_SECS`` constant (default 0.0) is awaited between
        the first miss and the retry; bump it if production shows a write-flush
        boundary that requires a bounded wait before the index catches up.

    Args:
        memory_service: Mem0 service with an async ``search`` method.
        project_id: Project scope forwarded to ``find_prior_memories``.
        task_id: Task identifier (str-coerced by ``find_prior_memories``).
        flag_type: Flag type; used in both the ``kind`` filter and the WARNING.
        run_id: Current run identifier; scoped into the kind filter so only
             the marker written by THIS run is returned (not stale priors).
        log: Logger to use (should be the ``flag_dedup`` module logger so
             caplog-based tests can capture WARNINGs under the right namespace).

    Returns:
        ``True`` if at least one matching marker is findable (initial search or
        retry); ``False`` if no match after retry or if an unexpected error
        occurred.  Within ``dedup_flags`` this drives the HIT-branch
        prior-deletion gate (skip if False) and the MISS-branch no-op WARNING.
    """
    try:
        query = _marker_query(task_id, flag_type)
        # run_id is included so that stale priors from earlier runs do NOT match.
        # Intentionally asymmetric with the pre-write dedup search (which omits
        # run_id to find priors from any earlier run).  The confirmation's job is
        # 'did MY write for THIS run land?' — a prior from an older run must not
        # masquerade as confirmation of the current write.
        kind = {'source': 'stage1_flag_marker', 'flag_type': flag_type, 'run_id': run_id}

        matches = await find_prior_memories(
            memory_service,
            project_id=project_id,
            task_id=task_id,
            kind=kind,
            query=query,
            categories=['observations_and_summaries'],
            limit=50,
            log=log,
        )
        if matches:
            return True

        # Miss on first attempt — log WARNING, wait the configured delay, then retry once.
        log.warning(
            'confirm_marker_persisted: marker not found after write for task %s'
            ' flag_type %s run_id %s — retrying search',
            task_id, flag_type, run_id,
        )
        await _sleep(_CONFIRM_RETRY_DELAY_SECS)
        retry_matches = await find_prior_memories(
            memory_service,
            project_id=project_id,
            task_id=task_id,
            kind=kind,
            query=query,
            categories=['observations_and_summaries'],
            limit=50,
            log=log,
        )
        if retry_matches:
            return True

        # Retry also missed — log final WARNING and return False.
        log.warning(
            'confirm_marker_persisted: could not confirm flag marker for task %s'
            ' flag_type %s run_id %s after retry — marker may be unfindable next cycle',
            task_id, flag_type, run_id,
        )
        return False
    except Exception as e:
        log.warning(
            'confirm_marker_persisted: unexpected error for task %s flag_type %s: %s',
            task_id, flag_type, e,
        )
        return False


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

    The search uses ``limit=501`` internally so that genuine overflow can be
    detected without false positives.  When exactly 500 results are returned
    Mem0 may have returned the entire set (no truncation); when 501 are
    returned it confirms more than 500 records exist and the excess is
    silently dropped.  A WARNING is logged in the overflow case so dashboards
    can alert on incomplete suppression coverage.  In practice suppression
    records are a small operator-managed set and 500 provides ample headroom.

    On search exception: logs a WARNING and returns *flags* unchanged
    (conservative pass-through — treats as "no suppression in effect").
    """
    if not flags:
        return []

    # Bulk fetch: bare query + filter by metadata post-hoc. See docstring re: limit=501.
    try:
        results = await memory_service.search(
            query='stage1_flag_suppression',
            project_id=project_id,
            categories=['observations_and_summaries'],
            stores=['mem0'],
            limit=501,
        )
    except Exception as e:
        logger.warning(
            'filter_suppressed search failed for project %s: %s', project_id, e
        )
        return flags

    if len(results) > 500:
        logger.warning(
            'filter_suppressed: result count exceeded 500 for project %s; '
            'suppression set truncated to 500',
            project_id,
        )
        results = results[:500]

    suppressed_task_ids: set[str] = set()
    for r in results:
        meta = r.metadata or {}
        if meta.get('kind') != 'stage1_flag_suppression':
            continue
        task_id = meta.get('task_id')
        if task_id is None or task_id == '':
            continue
        suppressed_task_ids.add(str(task_id))  # required: legacy records may carry task_id as int or str; coerce both sides for compat

    def _keep(f: dict[str, Any]) -> bool:
        flag_tid = f.get('task_id')
        if flag_tid is None or flag_tid == '':
            return True  # symmetric with producer-side suppression-record guard above
        return str(flag_tid) not in suppressed_task_ids

    return [f for f in flags if _keep(f)]


async def _write_and_confirm_marker(
    memory_service: Any,
    *,
    project_id: str,
    run_id: str,
    tid: str,
    ftype: str,
    log: logging.Logger,
    confirm_and_track,  # async callable: (response_memory_ids, active_miss_warning_msg, tripped_skip_warning_msg, *, tid, ftype) -> bool
    active_miss_warning_template: str,
    tripped_skip_warning_template: str,
) -> bool:
    """Write a stage1_flag_marker memory and confirm it is findable.

    Single source of truth for the canonical marker payload contract:
    - ``content``: ``f'Stage 1 flag marker: task={tid} type={ftype} from run={run_id}'``
    - ``category='observations_and_summaries'``
    - ``metadata={'source':'stage1_flag_marker', 'kind':'stage1_flag_marker',
                  'task_id':tid, 'flag_type':ftype,
                  'run_id':run_id, 'last_seen_run_id':run_id}``
    - ``_source='stage1_flag_dedup'`` sentinel

    **Validation guard (defense-in-depth):** before calling ``add_memory``,
    ``tid`` is checked by :func:`_is_valid_marker_task_id`.  Under normal
    operation this guard is never tripped — the early guard in
    :func:`dedup_flags` already validates ``tid`` before reaching this helper,
    so all real signatures (numeric, comma-joined, or canonical ``fp:+32-hex``)
    pass both guards.  This backstop exists for any future direct caller that
    bypasses the early guard.  When tripped for a genuinely-invalid ``tid``
    (e.g. ``'abc'``, malformed ``fp:`` variants, empty string) it logs a WARNING
    and returns ``False`` — ``add_memory`` and ``_confirm_and_track`` are NOT
    called.  Returning ``False`` (not raising) ensures:

    - On the HIT path, priors are NOT deleted (best-effort-replacement invariant:
      never delete priors when no replacement was written).
    - The confirmation circuit-breaker counter is untouched — a guard-skip for
      a genuinely-invalid ``tid`` is not a Mem0 brownout signal.

    On add_memory exception: logs a unified WARNING and returns ``False``.

    On success: delegates to ``confirm_and_track`` (the circuit-breaker-aware
    inner closure from ``dedup_flags``) and propagates its bool verbatim.

    The ``active_miss_warning_template`` is emitted by ``confirm_and_track``
    when a confirmation search was attempted but returned no result (ACTIVE
    breaker).  The ``tripped_skip_warning_template`` is emitted when
    confirmation was skipped because the breaker was already tripped and
    ``bool(response.memory_ids)`` was False (TRIPPED breaker).  Both templates
    are forwarded verbatim to ``confirm_and_track``.
    """
    # Defense-in-depth guard: reject genuinely-invalid task_id keys before writing
    # to Mem0.  Under normal operation the early guard in dedup_flags already validated
    # tid (numeric, comma-joined, or canonical fp:+32-hex), so this branch is not
    # reached for any real signature.  It stays cheap and silent for correct callers;
    # logging at WARNING here is appropriate because reaching this branch means an
    # unvalidated tid was passed directly to this helper — genuinely unexpected.
    # Returning False (not raising) preserves the HIT-path best-effort-replacement
    # invariant and keeps the circuit-breaker counter clean.
    if not _is_valid_marker_task_id(tid):
        log.warning(
            'flag_dedup: skipping stage1_flag_marker write for invalid task_id %r'
            ' flag_type %s — rejected by _is_valid_marker_task_id (defense-in-depth)',
            tid, ftype,
        )
        return False
    try:
        response = await memory_service.add_memory(
            content=f'Stage 1 flag marker: task={tid} type={ftype} from run={run_id}',
            category='observations_and_summaries',
            project_id=project_id,
            metadata={
                'source': 'stage1_flag_marker',
                'kind': 'stage1_flag_marker',
                'task_id': tid,
                'flag_type': ftype,
                'run_id': run_id,
                'last_seen_run_id': run_id,
            },
            causation_id=run_id,
            _source='stage1_flag_dedup',
        )
    except Exception as e:
        log.warning(
            'flag_dedup: failed to write marker for task %s flag_type %s: %s',
            tid, ftype, e,
        )
        return False
    # ``confirm_and_track`` is required to never raise (``confirm_marker_persisted``
    # has its own internal try/except; the breaker counter mutations are non-raising).
    # If that invariant is broken by a future refactor, the exception will propagate
    # out of this helper and abort the ``dedup_flags`` for-loop iteration.
    return await confirm_and_track(
        response.memory_ids,
        active_miss_warning_template,
        tripped_skip_warning_template,
        tid=tid, ftype=ftype,
    )


async def dedup_flags(
    memory_service: Any,
    project_id: str,
    run_id: str,
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate Stage 1 flagged items against prior ``stage1_flag_marker`` memories.

    For each flag in *flags*:

    - Signature is computed first via ``compute_flag_signature(flag)``
      (keyed on top-level task_id or cited_tasks fallback).  When that returns
      None, ``compute_content_fingerprint_signature(flag)`` is tried as a
      fallback (task-1654 Fix 2) for null-task_id flags lacking cited_tasks.
      Only when BOTH helpers return None is the flag returned unchanged — no
      I/O performed.
    - If a signature is computable, the resulting ``task_id`` (``tid``) is
      validated by ``_is_valid_marker_task_id``.  This guard accepts numeric
      keys, comma-joined integer lists, and canonical ``fp:+32-hex`` content-
      fingerprint keys (task-1670 Option A); it rejects only genuinely-invalid
      tids (empty string, malformed ``fp:`` variants, non-numeric strings, etc.).
      When rejected the flag is returned unchanged and no Mem0 I/O is performed.
    - For valid signatures (numeric, comma-joined, or canonical ``fp:``), Mem0
      is searched for a prior marker memory with matching ``task_id`` and
      ``flag_type``.  ``fp:``-keyed markers are a **Stage-1-internal dedup
      artifact**: they are never consumed or swept by Stage 2's
      ``_query_stage2_flags`` (which processes only ``flag_for_stage2=True``
      records) and are therefore safe to persist without triggering a Stage 2
      cleanup loop (task-1670 step-2 verification).
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

    # --- Confirmation circuit-breaker (task-1412) ---
    # Per-invocation counter: strictly consecutive miss count.  Reset to 0 on any
    # successful confirmation (True return).  When the count reaches
    # _CONFIRMATION_MISS_THRESHOLD, log ONE breaker WARNING and set
    # confirmation_disabled = True so the remainder of the batch skips
    # confirm_marker_persisted entirely and gates on bool(response.memory_ids).
    # Being function-local, these reset automatically at each dedup_flags call.
    # See: "Confirmation circuit-breaker (task-1412)" section in module docstring.
    consecutive_confirmation_misses: int = 0
    confirmation_disabled: bool = False

    async def _confirm_and_track(
        response_memory_ids: list[str],
        active_miss_warning_msg: str,
        tripped_skip_warning_msg: str,
        *,
        tid: str,
        ftype: str,
    ) -> bool:
        """Shared circuit-breaker helper for HIT and MISS branches.

        When the breaker is ACTIVE (``confirmation_disabled`` is False):
        - Calls ``confirm_marker_persisted``; on miss (``False`` return) emits
          ``active_miss_warning_msg``, increments the consecutive miss counter,
          and trips the breaker when the threshold is reached (one breaker WARNING
          logged at trip-time only).
        - On hit (``True`` return): resets the counter so sporadic misses don't
          accumulate.
        - Returns the bool from ``confirm_marker_persisted``.

        When the breaker is TRIPPED (``confirmation_disabled`` is True):
        - Skips ``confirm_marker_persisted``; gates on ``bool(response_memory_ids)``.
        - Emits ``tripped_skip_warning_msg`` iff ``bool(response_memory_ids)`` is False.
        - Returns ``bool(response_memory_ids)``.

        The two templates MUST be distinct so brownout logs cleanly separate genuine
        confirmation-miss flags (search attempted, returned no result) from gate-only
        flags (search skipped because breaker open).  This is the disambiguation
        contract operators rely on during a Mem0 brownout.

        ``tid`` and ``ftype`` are explicit keyword-only parameters so this helper
        is safe to schedule out-of-order (e.g. ``asyncio.gather``); the enclosing-
        loop variables would otherwise be captured by free-variable lookup,
        silently picking up the LAST iteration's values under concurrent scheduling.

        Mutates nonlocal ``consecutive_confirmation_misses`` and
        ``confirmation_disabled``; captures ``memory_service``, ``project_id``,
        ``run_id``, and ``logger`` from the enclosing ``dedup_flags`` scope.
        """
        nonlocal consecutive_confirmation_misses, confirmation_disabled
        if not confirmation_disabled:
            is_found = await confirm_marker_persisted(
                memory_service,
                project_id=project_id,
                task_id=tid,
                flag_type=ftype,
                run_id=run_id,
                log=logger,
            )
            if not is_found:
                # Both ``active_miss_warning_msg`` and ``tripped_skip_warning_msg``
                # MUST be printf-style strings with exactly two %s placeholders in
                # order: (task_id, flag_type).
                logger.warning(active_miss_warning_msg, tid, ftype)
                consecutive_confirmation_misses += 1
                if consecutive_confirmation_misses >= _CONFIRMATION_MISS_THRESHOLD:
                    logger.warning(
                        'flag_dedup: confirmation circuit-breaker tripped after %d'
                        ' consecutive misses; falling back to memory_ids gate for'
                        ' remainder of batch',
                        consecutive_confirmation_misses,
                    )
                    confirmation_disabled = True
            else:
                # Strictly consecutive: any successful confirmation resets the
                # counter so sporadic misses don't accumulate toward threshold.
                # Reset ONLY inside `if not confirmation_disabled` so a tripped
                # breaker can't be un-tripped mid-batch.
                consecutive_confirmation_misses = 0
            return is_found
        else:
            write_succeeded = bool(response_memory_ids)
            if not write_succeeded:
                logger.warning(tripped_skip_warning_msg, tid, ftype)
            return write_succeeded

    result: list[dict[str, Any]] = []
    for flag in flags:
        sig = compute_flag_signature(flag)
        # Content-fingerprint fallback (task-1654 Fix 2): for null-task_id flags
        # that lack cited_tasks, compute_flag_signature returns None.  Route them
        # through the content-fingerprint path so dedup_flags writes/matches a
        # marker and the finding stops re-escalating every cycle.
        # Only appended unchanged (pass-through) when BOTH helpers return None.
        if sig is None:
            sig = compute_content_fingerprint_signature(flag)
        if sig is None:
            result.append(flag)
            continue
        tid, ftype = sig
        # Guard: skip search + write for genuinely-invalid tids.
        # Canonical fp:+32-hex keys produced by compute_content_fingerprint_signature
        # PASS this guard (task-1670 Option A): they are Stage-1-internal dedup
        # artifacts that are safe to persist because Stage 2's _query_stage2_flags
        # processes only flag_for_stage2=True records and never touches stage1_flag_marker
        # rows regardless of task_id format.  Only truly-invalid tids (empty string,
        # malformed fp: variants, non-numeric/non-fp: strings) are rejected here and
        # logged at DEBUG (not a brownout signal, just an unexpected key shape).
        if not _is_valid_marker_task_id(tid):
            logger.debug(
                'flag_dedup: skipping stage1 dedup for invalid task_id %r'
                ' (flag_type %s) — rejected by _is_valid_marker_task_id',
                tid, ftype,
            )
            result.append(flag)
            continue
        # Delegate search+filter to the shared helper.  find_prior_memories logs a
        # WARNING under logger on search failure and returns [] so the else
        # branch below writes a fresh marker (best-effort on transient Mem0 outage).
        priors = await find_prior_memories(
            memory_service,
            project_id=project_id,
            task_id=tid,
            kind={'source': 'stage1_flag_marker', 'flag_type': ftype},
            query=_marker_query(tid, ftype),
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
            #     After writing, confirm the marker is findable via a read-back
            #     search (task-1400): add_memory may store content under a
            #     different id than the one returned.  write_succeeded is True
            #     only when the marker is confirmed findable by a subsequent search.
            #     An unconfirmed write (write exception OR confirmation miss)
            #     preserves priors for next cycle (best-effort at-least-one-marker).
            #
            #     _write_and_confirm_marker encapsulates the canonical payload,
            #     the try/except, and the delegation to _confirm_and_track (which
            #     encapsulates confirm_marker_persisted + circuit-breaker counter).
            # See: _confirm_and_track docstring and "Confirmation circuit-breaker"
            # section in the module docstring.
            write_succeeded = await _write_and_confirm_marker(
                memory_service,
                project_id=project_id, run_id=run_id, tid=tid, ftype=ftype, log=logger,
                confirm_and_track=_confirm_and_track,
                active_miss_warning_template=(
                    'flag_dedup: replacement marker for task %s flag_type %s could not'
                    ' be confirmed findable — skipping prior deletion'
                ),
                tripped_skip_warning_template=(
                    'flag_dedup: replacement marker for task %s flag_type %s —'
                    ' confirmation skipped (circuit-breaker open) and'
                    ' memory_ids gate failed —'
                    ' skipping prior deletion'
                ),
            )

            # (3) Delete ALL priors only if the new marker was confirmed FINDABLE
            #     (or, after circuit-breaker trip, if bool(response.memory_ids) is True).
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
            #
            # Post-write confirmation (task-1400): after writing, confirm the
            # marker is findable via a read-back search.  The WARNING is driven
            # off the bool return from _confirm_and_track (False = unfindable in
            # ACTIVE branch, or bool(response.memory_ids)==False in TRIPPED branch).
            #
            # Circuit-breaker (task-1412): _write_and_confirm_marker delegates to
            # _confirm_and_track (the same inner closure shared with the HIT branch),
            # so both branches share the same counter / disabled flag.
            # When the breaker is tripped, _confirm_and_track drives the "will not be
            # detected next cycle" WARNING off bool(response.memory_ids) instead.
            await _write_and_confirm_marker(
                memory_service,
                project_id=project_id, run_id=run_id, tid=tid, ftype=ftype, log=logger,
                confirm_and_track=_confirm_and_track,
                active_miss_warning_template=(
                    'flag_dedup: MISS marker for task %s flag_type %s could not be'
                    ' confirmed findable — recurring flag will not be detected next cycle'
                ),
                tripped_skip_warning_template=(
                    'flag_dedup: MISS marker for task %s flag_type %s —'
                    ' confirmation skipped (circuit-breaker open) and'
                    ' memory_ids gate failed —'
                    ' recurring flag will not be detected next cycle'
                ),
            )
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
    try:
        tid = int(task_id)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f'build_suppression_payload: task_id must be an int or numeric '
            f'string, got {task_id!r}'
        ) from e
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

    **cited_tasks fallback (PRD γ §9.3):** when the top-level ``task_id`` key
    is absent (``None``), the function derives a deterministic signature from the
    *sorted set* of all ``task_id`` values in ``cited_tasks``, comma-joined.
    This ensures multi-task findings produce the same signature regardless of
    citation order, and prevents two findings that share only the first cited
    task from colliding (reviewer finding dedup_correctness).  Callers that need
    precise single-task dedup should always set the top-level ``task_id``
    explicitly — the fallback is a best-effort heuristic for findings that omit
    it.

    Returns ``None`` for flags without enough signal to deduplicate — these are
    passed through unchanged by :func:`dedup_flags`.
    """
    task_id = flag.get('task_id')

    # Best-effort fallback: derive task_id from cited_tasks when the top-level
    # field is absent.  flag_type is still required at the top level.
    # Uses sorted(all task_ids) — not just the first — so multi-task findings
    # dedup deterministically regardless of citation order.
    if task_id is None:
        cited_tasks = flag.get('cited_tasks')
        if cited_tasks and isinstance(cited_tasks, list):
            task_ids = sorted(
                str(c['task_id'])
                for c in cited_tasks
                if isinstance(c, dict) and c.get('task_id') is not None
            )
            if task_ids:
                task_id = ','.join(task_ids)

    flag_type = flag.get('flag_type')
    if task_id is None or flag_type is None:
        return None
    return (str(task_id), str(flag_type))


# --------------------------------------------------------------------------- #
# Content-fingerprint helpers (task-1654 Fix 2)
# --------------------------------------------------------------------------- #

#: Sentinel flag_type used in the content-fingerprint (fp:…) signature when
#: the flag's own flag_type is None.  A stable string avoids a None value
#: breaking str-coercion in find_prior_memories / marker metadata writes.
#: Do NOT change without a marker migration — existing markers keyed by this
#: sentinel must remain findable by the new value.
_CONTENT_FP_FLAG_TYPE: str = '__content_fp__'


def _normalize_content_description(description: str) -> str:
    """Casefold + collapse internal whitespace (mirrors recon_report._normalize_description).

    A local copy avoids a server<-reconciliation import that would invert the
    package layering.  Both normalizers must stay aligned — if recon_report's
    implementation changes, update this one too.
    """
    return ' '.join(description.split()).casefold()


#: Prefix emitted by :func:`_content_fingerprint`.  Used as a single source of
#: truth so :func:`_is_valid_marker_task_id` and :func:`_content_fingerprint`
#: cannot drift apart silently.
_CONTENT_FP_PREFIX: str = 'fp:'

#: Number of hex characters kept from SHA-256 hexdigest (``digest[:32]``).
#: 128 bits of SHA-256 provides sufficient collision resistance for a dedup key
#: over recon findings.  Must match :func:`_content_fingerprint`'s slice length.
_CONTENT_FP_HEXLEN: int = 32

#: Compiled regex that matches ONLY canonical content-fingerprint marker keys:
#: ``fp:`` followed by exactly :data:`_CONTENT_FP_HEXLEN` lowercase hex digits.
#: Uppercase hex is excluded because :func:`hashlib.sha256().hexdigest` always
#: returns lowercase; accepting uppercase would widen the accept-set beyond what
#: the emitter can produce and introduce false positives.
_CONTENT_FP_RE: re.Pattern[str] = re.compile(
    rf'{re.escape(_CONTENT_FP_PREFIX)}[0-9a-f]{{{_CONTENT_FP_HEXLEN}}}\Z'
)


def _is_valid_marker_task_id(tid: str) -> bool:
    """Return True iff *tid* is a valid stage1_flag_marker key.

    Accepts:
    - A canonical content-fingerprint key: ``'fp:'`` followed by exactly
      :data:`_CONTENT_FP_HEXLEN` (32) lowercase hex digits, e.g.
      ``'fp:9216e85ac497b68d93043b64684eb049'``.  This is the ONLY shape
      emitted by :func:`_content_fingerprint`; the regex :data:`_CONTENT_FP_RE`
      enforces the exact length and character set so accept-pattern and
      emit-pattern cannot drift independently.
    - A bare non-negative integer string (e.g. ``'42'``, ``'0'``).
    - A comma-joined list of non-negative integers (e.g. ``'12,15'``), which is
      the shape produced by :func:`compute_flag_signature`'s ``cited_tasks``
      fallback for multi-task findings.

    Rejects:
    - Falsy / empty input.
    - Malformed fp: forms: ``'fp:'`` (no hex), too-short or too-long hex bodies,
      uppercase hex, non-hex characters in the body.
    - Any component that is not a non-negative integer after strip (numeric path).
    - Trailing/leading commas that yield empty components (e.g. ``'12,'``).

    Mirrors the codebase's canonical isdigit-based, dot-rejecting task-id
    convention (``_looks_like_task_id`` in task_interceptor.py and
    sqlite_task_backend.py) while additionally tolerating the comma-joined
    marker key and canonical fp: keys.  Defined as a LOCAL helper to avoid a
    server/middleware←reconciliation import inversion; see the local-copy
    convention in :func:`_normalize_content_description`.

    Pure, sync, no I/O.
    """
    if not tid:
        return False
    # Canonical content-fingerprint branch: fp: + exactly 32 lowercase hex chars.
    if _CONTENT_FP_RE.fullmatch(tid):
        return True
    # Numeric / comma-joined branch (existing convention, unchanged).
    components = tid.split(',')
    return all(part.strip().isdigit() for part in components)


def _content_fingerprint(description: str) -> str:
    """SHA-256 hex (first :data:`_CONTENT_FP_HEXLEN` chars) of the normalised description.

    Output format: :data:`_CONTENT_FP_PREFIX` + ``digest[:_CONTENT_FP_HEXLEN]``.
    Deterministic across processes and PYTHONHASHSEED (unlike builtin hash()).
    Truncation to :data:`_CONTENT_FP_HEXLEN` hex chars (128 bits of SHA-256) is
    sufficient collision resistance for a dedup key over recon findings.

    The emitted key is always accepted by :func:`_is_valid_marker_task_id` (the
    anti-drift invariant tested by ``TestIsValidMarkerTaskId.test_accepts_anti_drift_roundtrip``).
    """
    digest = hashlib.sha256(
        _normalize_content_description(description).encode('utf-8')
    ).hexdigest()
    return f'{_CONTENT_FP_PREFIX}{digest[:_CONTENT_FP_HEXLEN]}'


def compute_content_fingerprint_signature(
    flag: dict[str, Any],
) -> tuple[str, str] | None:
    """Return a content-fingerprint (fp:<hex>, flag_type_or_sentinel) or None.

    Activated ONLY when ALL of the following hold:
    - ``task_id`` is None (no top-level task anchor)
    - ``cited_tasks`` yields no task_id values (empty list, absent, or all None)
    - The normalized ``description`` is non-blank

    Returns None when any condition fails so callers can fall through to
    ``compute_flag_signature`` (which handles the cited_tasks path) or pass
    the flag through unchanged (when both helpers return None).

    When ``flag_type`` is None, the sentinel :data:`_CONTENT_FP_FLAG_TYPE` is
    used so the 2-tuple shape is preserved for marker write/match in
    ``dedup_flags``.

    Pure, sync, no I/O — safe to call from any context.
    """
    # Condition 1: top-level task_id must be None.
    if flag.get('task_id') is not None:
        return None

    # Condition 2: no usable task_id in cited_tasks.
    cited_tasks = flag.get('cited_tasks')
    if cited_tasks and isinstance(cited_tasks, list) and any(
        isinstance(c, dict) and c.get('task_id') is not None
        for c in cited_tasks
    ):
        return None

    # Condition 3: non-blank normalized description.
    description = flag.get('description') or ''
    if not _normalize_content_description(description):
        return None

    fp = _content_fingerprint(description)
    ftype = flag.get('flag_type') or _CONTENT_FP_FLAG_TYPE
    return (fp, str(ftype))


# --------------------------------------------------------------------------- #
# Absence guard helpers
# --------------------------------------------------------------------------- #

#: Flag types that assert a task is absent or phantom.  Flags of these types
#: must be validated by filter_false_absence_flags before Stage 2 can act on
#: them, because delete_memory is irreversible.
ABSENCE_FLAG_TYPES: frozenset[str] = frozenset({
    'task_absent',
    'phantom_task',
    'orphaned_knowledge',
})

#: Phrase produced by the sqlite backend when a task ID is not found.
#: Matched case-insensitively to tolerate minor message variations.
_NOT_FOUND_PHRASE: str = 'no tasks found for id'


async def filter_false_absence_flags(
    taskmaster: Any,
    project_root: str,
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop absence-asserting flags that cannot be positively confirmed absent.

    For each flag whose ``flag_type`` is in ``ABSENCE_FLAG_TYPES`` and that
    carries a ``task_id``, calls ``taskmaster.get_task(task_id, project_root)``
    and keeps the flag ONLY when ``confirm_task_absent`` returns ``True`` (task
    positively absent).  Flags that are present, inconclusive, or whose lookup
    raises are dropped — fail-closed, because delete_memory is irreversible.

    **Raised-exception path** (production RAW backend): The sqlite backend (and
    its TaskInterceptor middleware) RAISES ``TaskmasterError(
    'TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): N')`` on absence rather
    than returning a dict.  When get_task raises, the exception is normalised to
    ``{'error': str(exc), 'error_type': type(exc).__name__}`` and passed to
    ``confirm_task_absent``.  If that returns True (the not-found phrase is in
    ``str(exc)``), the flag is kept (task positively absent); otherwise it is
    dropped (fail-closed: TASKMASTER_UNAVAILABLE / timeout / generic raise →
    inconclusive → drop).

    Non-absence flags and absence flags without a ``task_id`` are passed through
    unchanged without issuing any get_task call.

    Degrades to a no-op pass-through when ``taskmaster`` or ``project_root`` is
    falsy (e.g. stage running without a configured Taskmaster backend).

    Structured drop observations are logged via
    ``logger.info('reconciliation.false_absence_flag_dropped', ...)`` with
    ``task_id`` and the reason (``'present'``, ``'inconclusive'``).

    Args:
        taskmaster: Object with an async ``get_task(task_id, project_root)``
            method, typically ``self.taskmaster`` in MemoryConsolidator.
        project_root: Project root path passed through to get_task.
        flags: List of flag dicts from Stage 1 ``items_flagged``.

    Returns:
        Filtered list with false-absence flags removed.
    """
    if not taskmaster or not project_root:
        return list(flags)

    async def _safe_get_task(task_id: Any) -> Any:
        """Fetch task with normalised exception handling.

        Returns the raw get_task result on success, or a normalised
        ``{'error': ..., 'error_type': ...}`` dict on any exception so that
        ``confirm_task_absent`` can classify both paths identically.
        """
        try:
            return await taskmaster.get_task(task_id, project_root)
        except Exception as exc:
            # Normalise: same dict shape as the MCP-wrapper path so
            # confirm_task_absent can classify the raised-exception path.
            return {'error': str(exc), 'error_type': type(exc).__name__}

    # Split flags into those requiring a get_task lookup and pass-throughs.
    # Track original position so the output list preserves input order.
    check_positions: list[int] = []  # indices of flags needing lookup
    check_task_ids: list[Any] = []

    for i, flag in enumerate(flags):
        flag_type = flag.get('flag_type')
        if flag_type in ABSENCE_FLAG_TYPES and flag.get('task_id') is not None:
            check_positions.append(i)
            check_task_ids.append(flag.get('task_id'))

    # Issue all get_task calls concurrently (typically only a handful per cycle).
    lookup_results: list[Any] = await asyncio.gather(
        *[_safe_get_task(tid) for tid in check_task_ids]
    )
    results_by_pos: dict[int, Any] = dict(zip(check_positions, lookup_results, strict=True))

    kept: list[dict[str, Any]] = []
    for i, flag in enumerate(flags):
        if i not in results_by_pos:
            # Non-absence flag or absence flag without task_id — pass through.
            kept.append(flag)
            continue

        result = results_by_pos[i]
        flag_type = flag.get('flag_type')
        task_id = flag.get('task_id')
        if confirm_task_absent(result):
            kept.append(flag)
        else:
            reason = 'present' if isinstance(result, dict) and 'error' not in result else 'inconclusive'
            logger.info(
                'reconciliation.false_absence_flag_dropped task_id=%s flag_type=%s reason=%s',
                task_id, flag_type, reason,
            )
            # drop: task is present or result is inconclusive

    return kept


def confirm_task_absent(get_task_result: object) -> bool:
    """Fail-closed classifier: True ONLY when get_task POSITIVELY confirms absence.

    Recognises the not-found signal produced by the SQLite task backend /
    get_task MCP wrapper: a dict where **both** of the following hold:

    * ``error_type == 'TaskmasterError'`` — tightens the match to the
      structured backend error class rather than relying on a phrase alone.
    * The ``error`` string contains 'No tasks found for ID(s)' (case-insensitive).

    Requiring the structured ``error_type`` reduces the risk of misclassifying
    an unrelated backend message that happens to embed the not-found phrase,
    while still matching the MCP-wrapper ``{error, error_type}`` dict and the
    normalised ``{'error': str(exc), 'error_type': type(exc).__name__}`` dict
    produced by filter_false_absence_flags for raised TaskmasterErrors.

    All other inputs — a valid task record, a generic/inconclusive error, None,
    an empty dict, or a non-dict value — return False (fail-closed).  The
    fail-closed contract is intentional: delete_memory is irreversible, so an
    inconclusive lookup must block deletion exactly like a present task.

    Args:
        get_task_result: The raw value returned by taskmaster.get_task() (or
            mcp__fused-memory__get_task).  Expected to be either a task dict
            (present) or an error dict (absent / inconclusive).

    Returns:
        True if and only if the result is a dict whose ``error_type`` is
        ``'TaskmasterError'`` and whose ``error`` string contains the canonical
        not-found phrase.  False in all other cases.
    """
    if not isinstance(get_task_result, dict):
        return False
    error = get_task_result.get('error')
    if not isinstance(error, str):
        return False
    error_type = get_task_result.get('error_type', '')
    return error_type == 'TaskmasterError' and _NOT_FOUND_PHRASE in error.lower()
