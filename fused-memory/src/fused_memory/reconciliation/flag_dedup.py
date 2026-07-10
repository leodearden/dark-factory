"""Flag deduplication helpers for Stage 1 (MemoryConsolidator).

This module provides code-level annotation of Stage 1's ``items_flagged``
output.  The LLM has no memory of prior cycles, so the same (task_id,
flag_type) pair can be emitted cycle after cycle.  Dedup/suppression/
acknowledge state is persisted in the ``recon_ledger`` SQLite-backed table
(:mod:`fused_memory.reconciliation.recon_ledger`), reached through
``memory_service.recon_ledger`` (task 2227).  The ledger is authoritative
and the only thing this module reads back — a best-effort Mem0 mirror
write accompanies every ledger write so legacy Mem0-based searchers keep
working during the cutover, but Mem0 is never searched by this module.

Note: this module does not suppress persistent flags before Stage 2 sees
them at the LLM level; suppression logic also lives in Stage 2's prompt
instructions, which direct the LLM to soft-handle annotated flags.  This
module enforces the suppression contract in code (see "Suppression"
below), making it authoritative over the prompt directive.

Ledger-backed marker UPSERT (task 2227)
----------------------------------------
``dedup_flags`` is the entry point.  For each flag with a computable
signature — ``(task_id, flag_type)`` via ``compute_flag_signature``,
falling back to ``compute_content_fingerprint_signature`` for null-task_id
flags lacking ``cited_tasks`` — the corresponding ``stage1_flag_marker``
row is looked up via ``get_by_identity(project_id, 'stage1_flag_marker',
task_id, flag_type, run_id='')``.  Note ``run_id=''`` in the identity: the
dedup identity is ``(task_id, flag_type)`` only, so the *current*
run_id/last_seen_run_id/deduped_against travel in ``payload_json``, not the
primary key.  If a prior row exists, its stored ``run_id`` becomes the
flag's ``persisted_from_run`` (the ``'unknown'`` sentinel is used on a
falsy/absent stored value); the row is then ``upsert``-ed with a fresh
payload and a self-refreshing 14-day ``expires_at`` TTL — every recurrence
pushes the expiry back out, so a still-recurring marker never ages out;
only a finding that stops recurring for 14 days is GC'd.

Because the identity excludes run_id, ``ON CONFLICT`` on the full primary
key guarantees **exactly one row survives** per (task_id, flag_type)
regardless of how many times the signature has recurred — across cycles
AND within a single ``dedup_flags`` call (a repeated signature later in
the same call reads back the row the earlier occurrence just committed).
This one primitive replaces the entire Mem0-era compensation chain
(task-1146/1165/1400/1412/1978): no separate search+write+confirm+delete
cycle, no post-write confirmation search, no confirmation circuit-breaker,
no in-batch ``seen_signatures`` memo, and no bounded reclamation limit —
the ledger's atomic UPSERT plus read-after-write consistency make all of
them unnecessary.  See ``dedup_flags`` for the full per-flag algorithm.

Suppression (task-1186, task-1966; ledger-backed as of task 2227)
--------------------------------------------------------------------
``dedup_flags`` calls ``filter_suppressed`` as its first step, before the
per-flag marker loop.  ``filter_suppressed`` reads
``memory_service.recon_ledger.list_suppressions(project_id)`` — one
indexed ``(project_id, record_kind, state)`` ledger query, no Mem0 search
— and drops flags matched by an active ``stage1_flag_suppression`` row. A
row with ``flag_type == ''`` is a WILDCARD (blanket-suppresses every
flag_type for its task_id); a row with a non-empty ``flag_type`` is
SCOPED to just that pair.  When both a wildcard and a scoped row exist for
the same task_id, the wildcard wins (union semantics — a blanket
suppression cannot be narrowed by a more specific record).
``write_suppression_record`` upserts these rows: ``flag_types=None``
writes a single blanket row; a non-empty list writes one scoped row per
flag_type.  Suppression rows never expire (``expires_at=None``) — they are
operator-managed and persist until explicitly cleared.  See
``filter_suppressed`` and ``write_suppression_record`` for full semantics.

Completion-marker same-cycle self-delete (task-2312)
-----------------------------------------------------
Some Stage 1 findings represent ONE-TIME completed/bookkeeping work (e.g.
"duplicate flag marker cleaned up", "dependency-parity gap resolved") that
will never recur, so persisting a marker for them would just orphan.  When
a flag carries ``flag_for_stage2`` present-and-explicitly-false (checked by
the pure helper ``_is_completion_flag`` — bool ``False`` or a
case-insensitive ``'false'`` string; absence of the key does not count),
``dedup_flags`` writes NO ``stage1_flag_marker`` row for it.  Instead it
sweeps any pre-existing prior for that exact (task_id, flag_type) via
``acknowledge_flag_marker(mode='delete')`` — now a ``mark_addressed`` call,
see "Acknowledge" below — and annotates the flag
``completion_marker_self_deleted=True`` (plus the usual
``last_seen_run_id``).  Removing the old in-batch ``seen_signatures`` memo
(see above) changes one accepted edge case: a duplicate completion
signature appearing more than once within a single ``dedup_flags`` call now
re-runs the sweep for each occurrence instead of memoizing after the
first — harmless, since ``mark_addressed`` is idempotent.  See
``dedup_flags`` and ``_is_completion_flag`` for the full branch.

Acknowledge (ledger-backed as of task 2227)
----------------------------------------------
``acknowledge_flag_marker`` flips a ``stage1_flag_marker`` row to
``state='addressed'`` via ``memory_service.recon_ledger.mark_addressed``,
returning ``1`` iff a matching row existed, else ``0`` — a no-op that can
never resurrect an unknown/already-GC'd signature.  The ledger has no
delete operation, so the Mem0-era ``mode='delete'`` vs ``mode='tag'``
distinction is obsolete: both collapse to the same ``mark_addressed`` call
(the ``mode`` parameter is retained only for call-site signature
compatibility).  ``acknowledge_resolved_flags`` is the batch entry point:
it computes and de-duplicates signatures, then fans out to
``acknowledge_flag_marker`` via ``gather_collect``, summing the counts.

Mem0 mirror (PRD decisions #4/#6: write-both, read-new)
-----------------------------------------------------------
Marker and suppression writes also perform a best-effort single
``add_memory`` mirror to Mem0 (wrapped in try/except; never raises, no
read-back, no confirmation search, no delete) so legacy Mem0-based
searchers keep working during the cutover to the ledger.  Reads in this
module NEVER consult Mem0 — the ledger is the sole read source.  When
``memory_service.recon_ledger`` is unset/``None`` (ledger disabled or not
yet wired), marker/suppression writes degrade to mirror-only and
``filter_suppressed``/``acknowledge_flag_marker``/``acknowledge_resolved_flags``
degrade to a conservative pass-through/no-op — this module never raises
because the ledger is absent.

Public API
----------
- ``compute_flag_signature(flag)`` — cheap, sync, no I/O.
- ``compute_content_fingerprint_signature(flag)`` — cheap, sync, no I/O;
  fallback signature for null-task_id flags lacking ``cited_tasks``.
- ``filter_suppressed(memory_service, project_id, flags)`` — async, one
  indexed ``recon_ledger`` query; drops suppressed flags before signature
  dedup.
- ``dedup_flags(memory_service, project_id, run_id, flags)`` — async, calls
  ``filter_suppressed`` first, then UPSERTs a ``stage1_flag_marker`` ledger
  row per (task_id, flag_type) signature; best-effort (exceptions are
  logged, not raised).  Flags explicitly marked ``flag_for_stage2=False``
  are instead treated as one-time completion markers (see above).
- ``write_suppression_record(memory_service, *, project_id, task_id,
  flag_types=None, causation_id=None)`` — async, upserts
  ``stage1_flag_suppression`` ledger row(s) for *task_id*.
- ``acknowledge_flag_marker(memory_service, *, project_id, run_id, task_id,
  flag_type, mode, log)`` — async, marks the ``stage1_flag_marker`` row for
  one (task_id, flag_type) signature ``state='addressed'``; never raises;
  returns the count acknowledged (0 or 1).
- ``acknowledge_resolved_flags(memory_service, project_id, run_id,
  resolved_flags, *, mode, log)`` — async, de-dupes signatures and fans out
  to ``acknowledge_flag_marker``; best-effort; returns the summed count.
- ``is_content_fingerprint_task_id(tid)`` — cheap, sync, no I/O; the fp:-only
  gate used to scope Gap 1/2 enrichment and sweep behaviour to fingerprint
  markers (task-2047).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NotRequired, TypedDict

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord
from fused_memory.utils.async_utils import gather_collect

logger = logging.getLogger(__name__)


class _SuppressionMetadata(TypedDict):
    """Producer-side contract: ``task_id`` is pinned to ``int``.

    Reader (``filter_suppressed``) tolerates and str-coerces both ``int``
    and ``str`` task_ids for backward compat with legacy hand-authored
    records — do NOT tighten the reader to int-only without a migration
    of any pre-existing str-task_id records in Mem0.

    ``flag_types`` is an OPTIONAL scoping allowlist (task-1966).  When
    present and non-empty, the record suppresses ONLY those flag_types for
    this task_id.  When absent — the legacy shape carried by all
    pre-existing hand-authored records — the record blanket-suppresses ALL
    flag_types for this task_id.
    """

    kind: Literal['stage1_flag_suppression']
    task_id: int
    flag_types: NotRequired[list[str]]


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
    """Drop flags matched by an active ``stage1_flag_suppression`` ledger record.

    Reads ``memory_service.recon_ledger.list_suppressions(project_id)`` -- an
    indexed ``(project_id, record_kind, state)`` query on the
    ``recon_ledger`` table (no Mem0 search) -- then builds a
    ``task_id -> (wildcard | flag_types)`` map from the returned rows'
    ``task_id``/``flag_type`` columns:

    - A row with ``flag_type == ''`` is a WILDCARD -- it blanket-suppresses
      every flag_type for that task_id.
    - A row with a non-empty ``flag_type`` is SCOPED -- it suppresses only
      that (task_id, flag_type) pair.
    - When both a scoped and a wildcard row exist for the same task_id
      (in either row order), the wildcard wins -- union semantics, since a
      blanket suppression cannot be narrowed by a more specific record.

    A flag is dropped iff its ``task_id`` has a wildcard entry, or has a
    scoped entry whose set contains the flag's (str-coerced) ``flag_type``.
    A flag with no ``flag_type`` (``None``/absent) can never match a scoped
    entry -- it is kept unless the task_id entry is a wildcard. The
    remaining flags are returned unchanged so they can proceed through the
    signature-dedup loop.

    Rows with ``task_id == ''`` are skipped when building the map (a
    degenerate/malformed suppression with no task target) -- this mirrors
    the old producer-side guard that rejected a missing/empty ``task_id``.

    Conservative pass-through (zero I/O beyond the attribute check itself):
    when *flags* is empty, or ``memory_service.recon_ledger`` is unset/
    ``None`` (ledger disabled or not yet wired), *flags* is returned
    unchanged -- the same "no suppression in effect" contract the old
    Mem0-search-exception path provided. The same pass-through applies, with
    a WARNING logged, when ``list_suppressions`` itself raises -- a ledger
    read failure must never abort the caller's whole ``dedup_flags`` batch.
    """
    if not flags:
        return []

    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        logger.debug(
            'filter_suppressed: no recon_ledger on memory_service for project %s; '
            'passing %d flag(s) through unfiltered',
            project_id,
            len(flags),
        )
        return flags

    try:
        rows = await ledger.list_suppressions(project_id)
    except Exception as e:
        logger.warning(
            'filter_suppressed: recon_ledger.list_suppressions failed for'
            ' project %s: %s (best-effort — treating as no suppression in'
            ' effect, passing %d flag(s) through unfiltered)',
            project_id,
            e,
            len(flags),
            exc_info=True,
        )
        return flags

    # task_id (str) -> None (wildcard/blanket) | set[str] (scoped flag_types
    # allowlist).  See docstring for wildcard-wins union semantics.
    suppressed: dict[str, set[str] | None] = {}
    for row in rows:
        tid_str = row.task_id
        if not tid_str:
            continue  # degenerate row with no task target; can never match a kept flag

        if tid_str in suppressed and suppressed[tid_str] is None:
            continue  # already wildcard for this task_id; cannot be narrowed further

        if not row.flag_type:
            # Wildcard/blanket row -> overrides any scoped entry accumulated
            # so far for this task_id (union semantics: wildcard wins).
            suppressed[tid_str] = None
            continue

        scoped = suppressed.get(tid_str)
        if not isinstance(scoped, set):
            scoped = set()
            suppressed[tid_str] = scoped
        scoped.add(row.flag_type)

    def _keep(f: dict[str, Any]) -> bool:
        flag_tid = f.get('task_id')
        if flag_tid is None or flag_tid == '':
            return True  # symmetric with producer-side suppression-record guard above
        tid_str = str(flag_tid)
        if tid_str not in suppressed:
            return True
        allowlist = suppressed[tid_str]
        if allowlist is None:
            return False  # wildcard/blanket suppression
        flag_type = f.get('flag_type')
        if flag_type is None:
            return True  # cannot match a scoped allowlist without a flag_type
        return str(flag_type) not in allowlist

    return [f for f in flags if _keep(f)]


# --------------------------------------------------------------------------- #
# Deduped-against UUID extraction (task-2047 Gap 1)
# --------------------------------------------------------------------------- #

#: Candidate flag-dict fields consulted by :func:`_extract_deduped_against_uuids`
#: for the resolvable Mem0 memory UUID(s) a duplicate-detection finding cites.
#: The canonical field is ``'deduped_against'`` — the field
#: ``prompts/stage1.py`` instructs the LLM to emit for duplicate-detection
#: findings (e.g. ``duplicate_procedural_knowledge``).  The remaining entries
#: are tolerated aliases: ``items_flagged`` entries are free-form LLM-emitted
#: dicts with no fixture pinning the exact field name in practice, so a small
#: union-of-candidates keeps extraction robust to shape drift.  Order only
#: affects readability here — extraction UNIONS values across every field.
#: This is a CONSCIOUS robustness/precision tradeoff: a generic alias (e.g.
#: ``'memory_ids'``) could in principle mean something other than "duplicate
#: of" on a producer's flag. To keep false-positive enrichment observable,
#: ``dedup_flags`` logs at INFO whenever a marker's ``deduped_against`` is
#: populated entirely from an alias rather than the canonical field.
_DEDUPED_AGAINST_FLAG_FIELDS: tuple[str, ...] = (
    'deduped_against',
    'duplicate_memory_ids',
    'duplicate_ids',
    'memory_ids',
    'cited_memory_ids',
)


def _extract_deduped_against_uuids(flag: dict[str, Any]) -> list[str]:
    """Return the sorted-unique memory UUID(s) *flag* cites as duplicates.

    Reads :data:`_DEDUPED_AGAINST_FLAG_FIELDS` in order and UNIONS every
    value found across all of them (a flag may carry more than one such
    field; all are honoured, not just the first present). For each field's
    value:

    - A bare ``str`` is treated as a single-item list (task-2047 step-1(c)).
    - A ``list`` is iterated element-by-element.
    - Any other shape (e.g. ``None``, ``int``, ``dict``) is skipped rather
      than raising, since ``flag`` is a free-form LLM-emitted dict.

    Each list element is type-checked before coercion: only ``str`` and
    ``int`` elements are accepted (e.g. an int ``123`` becomes ``'123'``);
    any other element shape (``dict``, nested ``list``, ``float``, ``bool``,
    etc.) is DROPPED rather than stringified, since ``str()`` on a
    dict/list would otherwise mint a junk 'UUID' like ``"{'k': 'v'}"``
    (task-2047 amendment).  Accepted elements are then stripped; blank
    results (empty string or whitespace-only) are also dropped.

    Returns ``[]`` when no candidate field is present, when every present
    field is empty, or when *flag* carries none of the candidate fields at
    all — never raises.

    Pure, sync, no I/O.
    """
    collected: set[str] = set()
    for field in _DEDUPED_AGAINST_FLAG_FIELDS:
        value = flag.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            candidates: list[Any] = [value]
        elif isinstance(value, list):
            candidates = value
        else:
            # Unexpected shape for this field — skip rather than raise; the
            # flag is a free-form LLM-emitted dict and other fields may still
            # yield a usable value.
            continue
        for item in candidates:
            if item is None:
                continue
            if not isinstance(item, (str, int)):
                # Structured/garbage element (dict, nested list, float, ...)
                # — drop rather than str()-coerce so it can't pollute
                # metadata.deduped_against with a non-UUID string.
                continue
            s = str(item).strip()
            if not s:
                continue
            collected.add(s)
    return sorted(collected)


def _is_completion_flag(flag: dict[str, Any]) -> bool:
    """Return True iff *flag* explicitly marks itself as ONE-TIME completed work.

    A flag is a completion marker iff ``flag['flag_for_stage2']`` is present
    AND explicitly false: the bool ``False``, or a string equal to ``'false'``
    case-insensitively (e.g. ``'false'``, ``'False'``, ``'FALSE'``).

    Absence of the key is DELIBERATELY excluded from this predicate — it must
    NOT be treated as a completion marker. The ``stage1_flag_marker`` rows
    ``dedup_flags`` upserts for recurring findings never set
    ``flag_for_stage2`` at all, and recurring-finding flags either omit the
    key or set it truthy. Treating absence as "false" would misclassify every
    ordinary recurring flag as a one-time completion, deleting the very
    marker cross-cycle dedup depends on and turning every cycle into a fresh
    MISS (infinite re-flagging). Only an EXPLICIT false value is a safe,
    unambiguous signal that the producer intends this finding to be a
    same-cycle completion marker rather than a recurring finding.

    Pure, sync, no I/O.
    """
    if 'flag_for_stage2' not in flag:
        return False
    value = flag['flag_for_stage2']
    if value is False:
        return True
    return isinstance(value, str) and value.strip().casefold() == 'false'


async def dedup_flags(
    memory_service: Any,
    project_id: str,
    run_id: str,
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Annotate Stage 1 flagged items against prior ``stage1_flag_marker`` ledger rows.

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
      When rejected the flag is returned unchanged and no ledger I/O is performed.
    - A flag explicitly marked ``flag_for_stage2=False`` (per
      :func:`_is_completion_flag`) is a ONE-TIME completion marker: see
      "Completion-marker same-cycle self-delete" above — no
      ``stage1_flag_marker`` row is written for it.
    - Otherwise, ``memory_service.recon_ledger`` (when attached) is read via
      ``get_by_identity(project_id, 'stage1_flag_marker', tid, ftype, run_id='')``
      — the dedup identity excludes ``run_id`` so this is always at most one
      row per ``(task_id, flag_type)``.  When a prior row exists, the flag is
      annotated ``persisted_from_run`` from its payload (``'unknown'`` sentinel
      on a falsy/absent stored ``run_id``).  ``last_seen_run_id`` is always set
      to the current ``run_id``.  The row is then ``upsert``-ed with a fresh
      payload/``created_at``/``expires_at`` (self-refreshing 14-day TTL) —
      ``ON CONFLICT`` on the full identity guarantees exactly one row survives
      regardless of how many times this signature has recurred, with no
      separate delete step and no dependence on read-after-write consistency
      across calls.
      The ``get_by_identity``/``json.loads``/``upsert`` sequence is wrapped in
      a single try/except: a ledger read/write failure (including a
      malformed/non-JSON ``payload_json`` on a prior row) is logged at
      WARNING and this flag simply gets no ledger annotation/persistence for
      this cycle — it never aborts the batch or propagates to the caller.
    - A best-effort Mem0 mirror (single ``add_memory``, no read-back/confirm/
      delete) is attempted after the ledger write — wrapped in try/except so a
      mirror failure never raises or rolls back the ledger write.  When no
      ledger is attached (``recon_ledger`` unset/``None``), this mirror write
      is the ONLY effect (pass-through dedup — no ``persisted_from_run`` can
      be computed without a ledger to read).

    ``persisted_from_run`` is set to the ``run_id`` stored in the prior row's
    payload.  If that field is absent, ``None``, or an empty string (i.e. any
    falsy value), the literal sentinel ``'unknown'`` is used instead.
    Downstream consumers (Stage 2 prompt, observability dashboards) can grep
    for ``'unknown'`` to detect malformed markers.

    Deduped-against enrichment (task-2047 Gap 1): for ``fp:``-keyed markers
    ONLY (per :func:`is_content_fingerprint_task_id`), the resolvable memory
    UUID(s) the flag cites as duplicates are extracted via
    :func:`_extract_deduped_against_uuids` and threaded into the ledger
    payload / Mem0 mirror metadata as ``deduped_against``.  Numeric and
    comma-joined markers always omit it — they are already resolvable anchors
    and their payload is unchanged.

    Returns the (possibly annotated) flag list.
    """
    # --- Authoritative suppression gate (task-1186) ---
    # Drop flags for tasks with active stage1_flag_suppression records BEFORE
    # the signature-dedup loop so suppressed flags never reach the per-flag
    # marker path.
    flags = await filter_suppressed(memory_service, project_id, flags)

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
        # Gap-1 enrichment (task-2047): scoped to fp:-keyed markers only — numeric
        # and comma-joined tids are already resolvable anchors and must not change
        # payload (see design decision in plan.json). Computed once per occurrence,
        # ahead of the ledger UPSERT below, so the ledger payload and the Mem0
        # mirror write share the identical value.
        deduped_against = (
            _extract_deduped_against_uuids(flag)
            if is_content_fingerprint_task_id(tid)
            else None
        )
        if deduped_against and not flag.get('deduped_against'):
            # Observability (task-2047 amendment): _DEDUPED_AGAINST_FLAG_FIELDS
            # unions several undocumented alias fields alongside the canonical
            # 'deduped_against' contract (prompts/stage1.py only instructs the
            # LLM to emit the canonical field). When enrichment is sourced
            # entirely from an alias — meaning the canonical field was absent
            # or empty — log it so a false-positive enrichment (an alias that
            # means something other than "duplicates", e.g. "all memories
            # examined") is observable rather than silently trusted.
            logger.info(
                'flag_dedup: deduped_against enrichment for task_id %r flag_type %s'
                ' sourced only from alias field(s), not the canonical'
                ' "deduped_against" field — verify producer output shape',
                tid, ftype,
            )
        # --- Completion-marker same-cycle self-delete (task-2312) ---
        # A flag the producer explicitly marks flag_for_stage2=False represents
        # ONE-TIME completed/bookkeeping work (e.g. a duplicate-marker cleanup or
        # a resolved dependency-parity gap) rather than a recurring finding that
        # must persist for cross-cycle dedup. Unlike the MISS/HIT paths below,
        # NO new stage1_flag_marker is written for it: a completion marker must
        # never survive the cycle that reported it, so there is nothing worth
        # persisting in the first place. This branch only SWEEPS any priors
        # accumulated by earlier cycles for this exact (task_id, flag_type)
        # signature via acknowledge_flag_marker(mode='delete') — self-healing
        # any pre-existing orphan for that signature (a no-op when none exist).
        #
        # amend (task-2312 review, revisited under the task-2227 ledger rewrite):
        # an earlier Mem0-backed version of this branch emitted (wrote and
        # confirmed) a marker and then immediately reclaimed it via
        # acknowledge_flag_marker in the SAME call. That acknowledge ran its own
        # independent Mem0 search rather than consuming the id the write just
        # returned, so under Mem0's read-after-write indexing lag the search
        # could race the just-written marker and miss it — leaving a fresh
        # orphan that (unlike the HIT path's collapse-to-one-row self-heal)
        # would never later collapse, because completion flags by definition
        # never recur. Skipping the write removes the race outright: there is
        # no new marker to lose track of, and any genuinely pre-existing prior
        # is still found and deleted below via the ledger's read-after-write-
        # consistent get_by_identity/mark_addressed.
        #
        # Dedup-safety: _is_completion_flag requires flag_for_stage2 to be
        # PRESENT and explicitly false. Absence (the shape every dedup
        # MISS/HIT marker and every ordinary recurring flag has) never matches,
        # so this branch is strictly additive and cannot regress cross-cycle
        # dedup for recurring findings — see _is_completion_flag's docstring.
        if _is_completion_flag(flag):
            # Best-effort, never raises (see acknowledge_flag_marker docstring).
            # Sweeps any priors left over from earlier cycles for this exact
            # (tid, ftype) signature; returns 0 (no-op) when none are found.
            await acknowledge_flag_marker(
                memory_service,
                project_id=project_id,
                run_id=run_id,
                task_id=tid,
                flag_type=ftype,
                mode='delete',
                log=logger,
            )
            flag = dict(flag)
            flag['completion_marker_self_deleted'] = True
            flag['last_seen_run_id'] = run_id
            # Idempotency (task-2312 review amendment, revised under the ledger
            # rewrite): a duplicate completion signature later in this same
            # batch simply re-runs this sweep — acknowledge_flag_marker's
            # ledger.mark_addressed is idempotent (re-addressing an
            # already-addressed row, or no-op'ing on an already-swept/absent
            # row), so no in-batch memo is needed to prevent it.
            result.append(flag)
            continue

        # --- Ledger-backed marker UPSERT (task 2227) ---
        # Dedup identity is (task_id, flag_type) with run_id='' in the ledger's
        # primary key, so a recurring signature always UPSERTs the SAME row —
        # no separate search+write+confirm+delete cycle, no dependence on
        # read-after-write consistency across calls or within one batch (a
        # repeat of this signature later in the SAME call reads back the row
        # just committed above).
        ledger = getattr(memory_service, 'recon_ledger', None)
        flag = dict(flag)
        persisted_from_run: str | None = None

        payload: dict[str, Any] = {
            'source': 'stage1_flag_marker',
            'kind': 'stage1_flag_marker',
            'task_id': tid,
            'flag_type': ftype,
            'run_id': run_id,
            'last_seen_run_id': run_id,
        }
        if deduped_against:
            payload['deduped_against'] = list(deduped_against)

        # Best-effort (module docstring / public-API contract): a ledger read
        # or write failure — including a malformed/non-JSON payload_json on a
        # prior row (legacy row, partial write, external corruption) — must
        # log and move on to the next flag rather than propagate and abort
        # the whole dedup_flags batch (memory_consolidator.run() calls this
        # with no surrounding try/except, so an unguarded raise here would
        # fail the entire Stage-1 run over a single bad row).
        if ledger is not None:
            try:
                prior = await ledger.get_by_identity(project_id, 'stage1_flag_marker', tid, ftype, '')
                if prior is not None:
                    prior_payload = json.loads(prior.payload_json)
                    persisted_from_run = prior_payload.get('run_id') or 'unknown'
                    if persisted_from_run == 'unknown':
                        logger.debug(
                            'flag_dedup: prior marker for task=%s flag_type=%s has malformed run_id metadata',
                            tid,
                            ftype,
                        )
                now = datetime.now(UTC)
                await ledger.upsert(ReconLedgerRecord(
                    project_id=project_id,
                    record_kind='stage1_flag_marker',
                    payload_json=json.dumps(payload),
                    state='active',
                    created_at=now.isoformat(),
                    task_id=tid,
                    flag_type=ftype,
                    run_id='',
                    expires_at=(now + timedelta(days=14)).isoformat(),
                ))
            except Exception as e:
                logger.warning(
                    'flag_dedup: recon_ledger read/write failed for marker'
                    ' task=%s flag_type=%s: %s (best-effort — flag still'
                    ' returned, no ledger annotation/persistence this cycle)',
                    tid,
                    ftype,
                    e,
                    exc_info=True,
                )

        if persisted_from_run is not None:
            flag['persisted_from_run'] = persisted_from_run
        flag['last_seen_run_id'] = run_id

        # Best-effort Mem0 mirror (PRD decision #4/#6 write-both/read-new): a
        # single add_memory, no read-back/confirm/delete loop.  Never raises —
        # a mirror failure must not roll back the ledger write above (or, when
        # no ledger is attached, must not abort the batch either).
        #
        # GC gap (task 2228 W5-κ, review finding robustness_unbounded_growth):
        # this Mem0-resident mirror (metadata.source='stage1_flag_marker') has
        # NO periodic collector. Task 2228 retired _sweep_stale_flag_markers —
        # its sole sweep — when marker GC collapsed onto
        # recon_ledger.ReconLedgerStore.gc() (see _gc_recon_markers above in
        # task_knowledge_sync.py), which deletes only ledger rows, never Mem0
        # memories. scripts/sweep_orphan_flag_markers.py still reads this same
        # source tag, but only as a manual, one-shot sweep for a disjoint
        # legacy-orphan concern (missing kind/task_id) — it does not collect
        # ordinary, well-formed, aging markers. No read path re-derives an
        # escalation count/threshold from this Mem0 pool (unlike
        # stage2_persistence_marker's _track_flag_persistence), so unbounded
        # growth here is a storage/latency cost, not a correctness regression
        # — the ledger row (above) is the read-of-record and IS reaped by
        # gc(). Accepted for the write-both/read-new cutover window; tracked
        # by follow-up task 2406 (retire this mirror write, or restore a
        # lightweight age sweep, once the cutover is confirmed complete).
        try:
            await memory_service.add_memory(
                content=f'Stage 1 flag marker: task={tid} type={ftype} from run={run_id}',
                category='observations_and_summaries',
                project_id=project_id,
                metadata=payload,
                causation_id=run_id,
                _source='stage1_flag_dedup',
            )
        except Exception as e:
            logger.debug(
                'flag_dedup: Mem0 mirror write failed for marker task=%s flag_type=%s: %s'
                ' (ledger write, if any, already committed)',
                tid, ftype, e,
                exc_info=True,
            )

        result.append(flag)
    return result


def build_suppression_payload(
    task_id: int | str, flag_types: list[str] | None = None
) -> SuppressionPayload:
    """Build the canonical ``stage1_flag_suppression`` Mem0 payload for *task_id*.

    Returns a :class:`SuppressionPayload` with ``content``, ``category``, and
    ``metadata`` fields matching the canonical schema documented in the
    ``## Flag Suppression Check`` section of ``STAGE1_SYSTEM_PROMPT``.
    ``task_id`` is coerced to ``int`` so the producer always pins the integer
    type regardless of how the caller obtained the id.

    ``project_id`` is intentionally absent — it is a write-time concern that
    must be passed separately to ``memory_service.add_memory``, keeping this
    helper pure and reusable across projects.

    ``flag_types`` is an OPTIONAL scoping allowlist (task-1966).  When a
    non-empty list is given, each element is coerced to ``str`` and the list
    is sorted+deduped before being stored under ``metadata.flag_types`` — the
    record then suppresses ONLY those (task_id, flag_type) pairs.  When
    ``None`` or empty (the default), ``metadata.flag_types`` is omitted
    entirely and the record keeps the legacy blanket-suppression-for-task_id
    shape, so ``build_suppression_payload(task_id)`` is unchanged for all
    existing callers.

    Canonical schema (Mem0, observations_and_summaries category):
      - ``metadata.kind = "stage1_flag_suppression"``
      - ``metadata.task_id = <N>`` (int — coerced by this function)
      - ``metadata.flag_types = [<str>, ...]`` (optional; sorted-unique)
      - ``content = "STAGE 1 FLAG SUPPRESSION task_id=<N>"``
    """
    try:
        tid = int(task_id)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f'build_suppression_payload: task_id must be an int or numeric '
            f'string, got {task_id!r}'
        ) from e
    metadata: _SuppressionMetadata = {
        'kind': 'stage1_flag_suppression',
        'task_id': tid,
    }
    if flag_types:
        metadata['flag_types'] = sorted({str(ft) for ft in flag_types})
    return {
        'content': f'STAGE 1 FLAG SUPPRESSION task_id={tid}',
        'category': 'observations_and_summaries',
        'metadata': metadata,
    }


async def write_suppression_record(
    memory_service: Any,
    *,
    project_id: str,
    task_id: int | str,
    flag_types: list[str] | None = None,
    causation_id: str | None = None,
) -> AddMemoryResponse:
    """Upsert a ``stage1_flag_suppression`` record to the ledger for *task_id*.

    Builds the canonical payload via :func:`build_suppression_payload` (which
    coerces *task_id* to ``int``, validates it, and pins ``metadata.kind``/
    ``content``) then upserts one ``recon_ledger`` row per entry in
    ``(flag_types or [''])`` to ``memory_service.recon_ledger`` — ``''`` is
    the blanket/wildcard ``flag_type`` (suppresses every flag_type for
    *task_id*); a non-empty ``flag_types`` list upserts one SCOPED row per
    flag_type. Each row's identity is ``(project_id,
    'stage1_flag_suppression', task_id, flag_type, run_id='')``, so a
    repeated call with the same arguments UPSERTs the same row(s) — the
    suppression row count never grows on recurrence. ``expires_at=None``:
    suppressions are operator-managed and never expire via TTL.

    ``memory_service.recon_ledger`` being unset/``None`` (ledger disabled or
    not yet wired) skips the ledger write entirely — this degrades to a
    Mem0-only mirror write, matching :func:`filter_suppressed`'s
    pass-through contract when it finds no ledger to read.

    After the ledger write(s), best-effort mirrors the same payload to Mem0
    via ``memory_service.add_memory`` (PRD decision #4/#6 write-both/
    read-new) — wrapped in try/except so a Mem0 failure never raises past
    this function; reads never consult Mem0. On a mirror failure (or when
    the ledger write happened but the mirror wasn't attempted for another
    reason) an empty :class:`AddMemoryResponse` is synthesized so the return
    type stays uniform.

    The ``_source='stage1_flag_suppression'`` sentinel distinguishes these
    mirror writes from ``'stage1_flag_dedup'`` and ``'targeted_recon'``
    writes in the audit journal, enabling per-class retention and query
    filtering.

    ``flag_types`` is an OPTIONAL scoping allowlist (task-1966), forwarded
    verbatim to :func:`build_suppression_payload` for the mirror payload.
    When a non-empty list is given, the record suppresses ONLY those
    (task_id, flag_type) pairs.  When ``None`` or empty (the default),
    ``metadata.flag_types`` is omitted from the mirror payload and the
    ledger gets a single blanket (``flag_type=''``) row.

    Canonical schema (Mem0 mirror, observations_and_summaries category):
      - ``metadata.kind = "stage1_flag_suppression"``
      - ``metadata.task_id = <N>`` (int — coerced by build_suppression_payload)
      - ``metadata.flag_types = [<str>, ...]`` (optional; sorted-unique)
      - ``content = "STAGE 1 FLAG SUPPRESSION task_id=<N>"``

    Returns the :class:`AddMemoryResponse` from the memory service so callers
    can inspect ``memory_ids`` for empty-list deduplication / no-op detection.
    """
    payload = build_suppression_payload(task_id, flag_types=flag_types)
    tid = str(payload['metadata']['task_id'])

    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is not None:
        now_iso = datetime.now(UTC).isoformat()
        payload_json = json.dumps(payload['metadata'])
        for ft in flag_types or ['']:
            await ledger.upsert(ReconLedgerRecord(
                project_id=project_id,
                record_kind='stage1_flag_suppression',
                payload_json=payload_json,
                state='active',
                created_at=now_iso,
                task_id=tid,
                flag_type=str(ft),
                run_id='',
                expires_at=None,
            ))

    try:
        return await memory_service.add_memory(
            **payload,
            project_id=project_id,
            causation_id=causation_id,
            _source='stage1_flag_suppression',
        )
    except Exception:
        logger.debug(
            'write_suppression_record: Mem0 mirror write failed for task_id=%s '
            'project_id=%s (ledger write, if any, already committed)',
            tid,
            project_id,
            exc_info=True,
        )
        return AddMemoryResponse(memory_ids=[])


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
#: breaking str-coercion in ledger identity columns / marker metadata writes.
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


def is_content_fingerprint_task_id(tid: str) -> bool:
    """Return True iff *tid* is a canonical content-fingerprint marker key.

    This is the fp:-ONLY gate: ``'fp:'`` followed by exactly
    :data:`_CONTENT_FP_HEXLEN` (32) lowercase hex digits — the single shape
    emitted by :func:`_content_fingerprint`.  Unlike
    :func:`_is_valid_marker_task_id` (which ALSO accepts bare numeric and
    comma-joined tids as valid marker keys), this helper rejects every shape
    other than the canonical fp: key, including otherwise-valid numeric and
    comma-joined marker keys.

    Public (no leading underscore) because it is imported by
    ``task_knowledge_sync`` (task-2047 Gap 2) to scope the cross-cycle
    fingerprint-marker sweep to fp:-keyed markers only, leaving numeric
    markers on the existing 14-day age-only GC.

    Pure, sync, no I/O.
    """
    return bool(tid) and bool(_CONTENT_FP_RE.fullmatch(tid))


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
# Stale count-snapshot correction filter (task-1786)
# --------------------------------------------------------------------------- #

#: Per-cycle task-count drift bound used by filter_stale_count_snapshot_corrections.
#:
#: Rationale: in normal operation the authoritative ## Active Task Tree header advances
#: by at most one task between consecutive snapshot writes (one task created or
#: completed per recon cycle).  The incident (run 929b4135 finding 2ebc814c) showed
#: a drift of exactly +1/+1, consistent with one task-creation event between the edge
#: write and the Stage-1 LLM read.  This constant bounds the "stale but correct" zone:
#: a componentwise delta ≤ 1 on a monotonically-increasing snapshot pair is explained
#: by normal task-count churn and must NOT be treated as a data-integrity error.
#:
#: Widen ONLY if operational evidence shows that cycles routinely create or complete
#: more than one task between snapshot writes — today's cadence does not justify >1.
STALE_SNAPSHOT_CADENCE_DELTA: int = 1

#: Correction-language triggers used by filter_stale_count_snapshot_corrections.
#: A flag's combined description+suggested_action text must contain at least one of
#: these substrings (or match the word-boundary 'incorrect' regex below) to qualify
#: as a potential correction finding.
#:
#: 'correct' alone is intentionally EXCLUDED: "snapshot X is correct" must NOT trigger
#: the gate.  'incorrect' is included via a word-boundary regex so that 'is correct'
#: does not fire.
_CORRECTION_LANGUAGE_SUBSTRINGS: tuple[str, ...] = (
    'off by',
    'off-by',
    'should be',
    'should read',
    'should now be',
    'corrected to',
    'is wrong',
    'actual count',
)

#: Word-boundary regex for 'incorrect' — matches the word 'incorrect' at a word
#: boundary, case-insensitively.  NOTE: this WILL match 'incorrect' inside the
#: phrase 'not incorrect' (the regex has no lookbehind exclusion for 'not').
#: That phrasing is vanishingly rare in LLM finding text, so the practical impact
#: is negligible; but the comment here is intentionally accurate about the behaviour.
#: The regex critically does NOT fire on bare 'is correct' because 'correct' alone
#: lacks the 'in' prefix — the \b boundary is anchored on the full word 'incorrect'.
_INCORRECT_WORD_RE: re.Pattern[str] = re.compile(r'\bincorrect\b', re.IGNORECASE)

#: Count-group regex: matches ≥2 integers joined by separators that appear in
#: task-count snapshot strings.  The separators allow optional status words
#: (done|cancelled|pending|in-progress|blocked|deferred|review|total|merge-deferred)
#: between the integers, mirroring the lexicon in task_filter.COUNT_SNAPSHOT_RE.
#:
#: NOTE: this is a LOCAL copy of the snapshot-detection lexicon from
#: task_filter.COUNT_SNAPSHOT_RE to honour the no-import-inversion convention
#: (reconciliation must not import from middleware).  If task_filter's status-word
#: list ever changes, update this regex accordingly.
#:
#: The pattern requires at least 2 integers (arity≥2) so that stray single-digit
#: numerals (e.g. the '1' in 'off by 1') are structurally excluded from being
#: treated as a count-pair operand.
_COUNT_GROUP_RE: re.Pattern[str] = re.compile(
    r'\d+'                                           # first integer
    r'(?:'                                           # separator group (non-capturing)
    r'[\s,/]+'                                       # plain separator: space, comma, slash
    r'(?:'                                           # optional status-word interleave
    # Status-word alternation kept aligned with task_filter.COUNT_SNAPSHOT_RE:
    #   cancell?ed  — matches both 'canceled' (US) and 'cancelled' (UK)
    #   in[-_ ]?progress — matches 'in-progress', 'in_progress', 'in progress'
    #   merge[-_ ]?deferred — matches 'merge-deferred', 'merge_deferred', 'merge deferred'
    r'(?:done|cancell?ed|pending|in[-_ ]?progress|blocked|deferred|review|total|merge[-_ ]?deferred)'
    r'[\s,/]+'
    r')?'
    r'\d+'                                           # subsequent integer
    r')+'                                            # one or more additional integer slots
,
    re.IGNORECASE,
)


def _extract_count_groups(text: str) -> list[tuple[int, ...]]:
    """Extract count-groups of arity ≥2 from *text*.

    Returns a list of tuples, each containing the integers found in one matched
    count-group.  Only groups with arity ≥ 2 (i.e. at least two integers) are
    returned; single-integer matches are structurally excluded by _COUNT_GROUP_RE.

    Example:
        '634 done / 607 total but should be 635 done / 608 total' →
        [(634, 607), (635, 608)]
    """
    groups: list[tuple[int, ...]] = []
    for match in _COUNT_GROUP_RE.finditer(text):
        integers = tuple(int(n) for n in re.findall(r'\d+', match.group()))
        if len(integers) >= 2:
            groups.append(integers)
    return groups


def _has_correction_language(text: str) -> bool:
    """Return True iff *text* contains at least one correction-language trigger."""
    lowered = text.lower()
    if any(sub in lowered for sub in _CORRECTION_LANGUAGE_SUBSTRINGS):
        return True
    return bool(_INCORRECT_WORD_RE.search(text))


def filter_stale_count_snapshot_corrections(
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop flags that are false 'off-by-N correction' findings on stale task-count snapshot edges.

    A flag is DROPPED iff all three conditions hold:
      (a) The combined ``description`` + ``suggested_action`` text contains
          correction language (fixed lexicon: 'off by', 'off-by', 'should be',
          'should read', 'should now be', 'corrected to', 'is wrong', 'actual count',
          or the word 'incorrect' at a word boundary — but NOT bare 'correct').
      (b) At least two count-groups of arity ≥ 2 are extractable from the combined
          text (paired snapshots like '634/607' or '634 done / 607 total').
          Requiring arity ≥ 2 structurally excludes stray digits like the '1' in
          'off by 1' from becoming a comparison operand.
      (c) After order-preserving deduplication, the combined text yields **exactly
          two distinct** arity-≥2 count-groups (if three or more distinct groups are
          found the flag is KEPT — a clean stale-drift correction references exactly
          two distinct numeric snapshots, though the proposed value may appear more
          than once across ``description`` and ``suggested_action``).  The two
          groups (treated as current and proposed) have equal arity, proposed ≥
          current componentwise (monotonic drift), and the maximum componentwise
          delta ≤ :data:`STALE_SNAPSHOT_CADENCE_DELTA`.

    The "exactly two distinct groups" constraint in condition (c) is intentional:
    if a flag's text contains three or more *distinct* arity-≥2 count-groups, the
    positional current/proposed identification (unique_groups[0] and unique_groups[1])
    could be confused by an incidental near-equal pair appearing before a genuine
    large-discrepancy pair.  Bailing to KEEP for these ambiguous texts avoids that
    failure mode while accommodating the common pattern where the proposed value is
    restated in both the description and the suggested_action.

    Otherwise the flag is KEPT (fail-open).  This conservative posture ensures that
    large discrepancies, count DECREASES, arity mismatches, reversed-order phrasings,
    and flags without extractable snapshot pairs are never silently discarded.

    This is the third (finding-side) layer of the snapshot-discipline defense:
    - Layer 1 (input-side): ``strip_snapshot_lines`` / ``is_count_snapshot`` in
      ``task_filter.py`` strips count-snapshot lines from the pre-assembled payload.
    - Layer 2 (write-side): ``ReconSnapshotWriteRejected`` server guard in
      ``server/tools.py`` blocks ``recon-stage-*`` agents from writing
      ``temporal_facts`` count-snapshot edges.
    - Layer 3 (this function): post-processor over ``items_flagged`` that drops
      findings whose text matches the stale-by-design oscillation signature.

    The first two layers miss the finding because the Stage 1 LLM can discover stale
    snapshot edges via its own live ``search``/``get_entity`` calls mid-run; this
    filter catches those findings before they reach ``dedup_flags`` and write a
    ``stage1_flag_marker`` or trigger a Stage 2 action.

    Pure, sync, no I/O — safe to call from any context.

    Args:
        flags: List of flag dicts from Stage 1 ``items_flagged``.

    Returns:
        Filtered list with false stale-snapshot-correction flags removed.
    """
    kept: list[dict[str, Any]] = []
    for flag in flags:
        description = flag.get('description') or ''
        suggested_action = flag.get('suggested_action') or ''
        combined = f'{description} {suggested_action}'.strip()

        # Condition (a): correction language present?
        if not _has_correction_language(combined):
            kept.append(flag)
            continue

        # Condition (b): ≥2 count-groups of arity ≥2 extractable?
        groups = _extract_count_groups(combined)
        if len(groups) < 2:
            kept.append(flag)
            continue

        # Condition (b, cont.): deduplicate groups (order-preserving) then require
        # EXACTLY two DISTINCT groups.  The same proposed value often appears in both
        # description ("should be 635/608") and suggested_action ("correct to 635/608"),
        # so naive len(groups) can be 3 for a clean stale-drift correction.  After
        # deduplication, a clean correction always has exactly 2 distinct groups.
        # Any other count (0, 1, or ≥3 distinct groups) means the text is degenerate
        # or ambiguous — bail to KEEP (fail-open).  Only when len==2 is a well-defined
        # positional current/proposed pair guaranteed; any other count risks:
        #   len==1: proposed restated identically in both fields → only one group, no
        #           "current" to compare against (was IndexError pre-fix)
        #   len≥3:  ambiguous text where positional groups[0]/groups[1] might pair an
        #           incidental near-equal prefix with a later large-discrepancy mention
        seen: set[tuple[int, ...]] = set()
        unique_groups: list[tuple[int, ...]] = []
        for g in groups:
            if g not in seen:
                seen.add(g)
                unique_groups.append(g)
        if len(unique_groups) != 2:
            kept.append(flag)
            continue

        current, proposed = unique_groups[0], unique_groups[1]

        # Condition (c): equal arity, monotonic, delta ≤ STALE_SNAPSHOT_CADENCE_DELTA?
        if len(current) != len(proposed):
            kept.append(flag)
            continue

        deltas = [p - c for c, p in zip(current, proposed, strict=True)]
        # Not monotonic (any decrease) → KEEP as potential integrity finding
        if any(d < 0 for d in deltas):
            kept.append(flag)
            continue

        # Delta too large → KEEP as potential integrity finding
        if max(deltas) > STALE_SNAPSHOT_CADENCE_DELTA:
            kept.append(flag)
            continue

        # All conditions met → DROP (stale-by-design, not erroneous)
        logger.debug(
            'filter_stale_count_snapshot_corrections: dropping stale snapshot correction flag '
            'task_id=%s flag_type=%s current=%s proposed=%s max_delta=%d',
            flag.get('task_id'), flag.get('flag_type'), current, proposed, max(deltas),
        )
        # do NOT append to kept — flag is dropped

    return kept


# --------------------------------------------------------------------------- #
# Terminal-metadata guard helpers (task-1725)
# --------------------------------------------------------------------------- #

#: Flag types that assert a task has stale / left-over metadata blobs.
#: Both spellings are included to be robust against LLM naming drift.
STALE_METADATA_FLAG_TYPES: frozenset[str] = frozenset({
    'stale_metadata',
    'task_metadata_stale',
})

#: Task statuses that represent terminal states with no further execution need.
#: A task in one of these states will never re-execute, so its metadata blobs
#: have no execution-time consumer and stale_metadata flags for it are noise.
#:
#: Deliberately excludes ``'deferred'`` and ``'blocked'``: although the steward
#: treats those as terminal decisions, a deferred or blocked task *may* resume
#: and still have live execution-time need for its metadata.  Add them here only
#: if it is confirmed that deferred/blocked tasks are permanently non-executable
#: in this deployment.
TERMINAL_STATUSES: frozenset[str] = frozenset({
    'cancelled',
    'done',
})


def _extract_terminal_status(result: object) -> str:
    """Extract task status from a get_task result, mirroring task_interceptor._extract_status.

    Checks top-level ``status`` first, then ``data['status']``, else returns
    ``'unknown'``.  Returns ``'unknown'`` for any non-dict input.

    This is a local copy of the extraction logic to honour the no-import-inversion
    convention (reconciliation must not import from middleware).

    **Sibling copies** — keep in sync if the get_task response shape ever changes:

    * ``middleware/task_interceptor._extract_status`` (~line 3292) — canonical source
    * ``reconciliation/stages/task_knowledge_sync._extract_status`` (~line 57) —
      same logic but assumes a dict input (no non-dict guard)
    """
    if not isinstance(result, dict):
        return 'unknown'
    status = result.get('status')
    if isinstance(status, str) and status:
        return status
    data = result.get('data')
    if isinstance(data, dict):
        nested = data.get('status')
        if isinstance(nested, str) and nested:
            return nested
    return 'unknown'


async def filter_terminal_metadata_flags(
    taskmaster: Any,
    project_root: str,
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop stale_metadata flags for tasks that are in a terminal state.

    For each flag whose ``flag_type`` is in ``STALE_METADATA_FLAG_TYPES`` and
    that carries a ``task_id``, calls ``taskmaster.get_task(task_id,
    project_root)`` and DROPS the flag iff the extracted status is in
    ``TERMINAL_STATUSES`` (``'cancelled'`` or ``'done'``).

    **Fail-safe direction**: this filter DROPS flags, so it drops ONLY on
    positively-confirmed terminal status.  get_task errors, non-dict results,
    ``'unknown'`` status, or any non-terminal status => KEEP the flag.  This
    is the conservative default: a transient get_task failure costs at most one
    extra dedup cycle and self-heals next cycle.

    Non-stale-metadata flags and stale-metadata flags without a ``task_id``
    are passed through unchanged without any get_task call.

    Degrades to a no-op pass-through when ``taskmaster`` or ``project_root`` is
    falsy (mirrors filter_false_absence_flags).

    Args:
        taskmaster: Object with an async ``get_task(task_id, project_root)``
            method, typically ``self.taskmaster`` in MemoryConsolidator.
        project_root: Project root path passed through to get_task.
        flags: List of flag dicts from Stage 1 ``items_flagged``.

    Returns:
        Filtered list with stale_metadata flags for terminal tasks removed.
    """
    if not taskmaster or not project_root:
        return list(flags)

    # Split flags into those requiring a get_task lookup and pass-throughs.
    check_positions: list[int] = []
    check_task_ids: list[Any] = []

    for i, flag in enumerate(flags):
        flag_type = flag.get('flag_type')
        if flag_type in STALE_METADATA_FLAG_TYPES and flag.get('task_id') is not None:
            check_positions.append(i)
            check_task_ids.append(flag.get('task_id'))

    # Detect potential LLM naming drift: flag_type strings that look like
    # stale-metadata variants (contain 'stale') but are not in
    # STALE_METADATA_FLAG_TYPES.  When the model emits an unrecognised spelling
    # the filter silently becomes a no-op; this log makes that observable so
    # operators can update STALE_METADATA_FLAG_TYPES.
    drift_candidates = [
        ft
        for flag in flags
        if (ft := flag.get('flag_type')) is not None
        and isinstance(ft, str)
        and 'stale' in ft.lower()
        and ft not in STALE_METADATA_FLAG_TYPES
    ]
    if drift_candidates:
        logger.info(
            'reconciliation.terminal_metadata_filter_possible_drift '
            'unmatched_flag_types=%s known_types=%s '
            '— update STALE_METADATA_FLAG_TYPES if drift confirmed',
            drift_candidates,
            sorted(STALE_METADATA_FLAG_TYPES),
        )

    if not check_positions:
        return list(flags)

    async def _safe_get_task(task_id: Any) -> Any:
        try:
            return await taskmaster.get_task(task_id, project_root)
        except Exception as exc:
            logger.debug(
                'reconciliation.terminal_metadata_filter_get_task_error task_id=%s error=%s',
                task_id, exc,
            )
            return None  # KEEP flag on error (fail-safe)

    lookup_results: list[Any] = await asyncio.gather(
        *[_safe_get_task(tid) for tid in check_task_ids]
    )
    results_by_pos: dict[int, Any] = dict(zip(check_positions, lookup_results, strict=True))

    kept: list[dict[str, Any]] = []
    for i, flag in enumerate(flags):
        if i not in results_by_pos:
            kept.append(flag)
            continue

        result = results_by_pos[i]
        status = _extract_terminal_status(result)
        task_id = flag.get('task_id')
        flag_type = flag.get('flag_type')

        if status in TERMINAL_STATUSES:
            logger.info(
                'reconciliation.terminal_metadata_flag_dropped task_id=%s status=%s',
                task_id, status,
            )
            # drop: task is terminal; metadata blobs have no execution-time consumer
        else:
            kept.append(flag)

    return kept


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


# --------------------------------------------------------------------------- #
# Blocked-snapshot finding filter for Stage 3 (task-1840)
# --------------------------------------------------------------------------- #

#: Categories that are subject to blocked-snapshot suppression.  Only flags
#: in these categories whose text matches the task-count-snapshot signature are
#: dropped; all other categories pass through unchanged (fail-open).
_SUPPRESSED_SNAPSHOT_CATEGORIES: frozenset[str] = frozenset({
    'missing_knowledge',
    'memory_stale',
})

#: Case-insensitive marker substrings that identify a finding as being about a
#: task-count snapshot temporal_fact edge.  The list targets the absence-wording
#: shape used by Stage-3 LLM findings (no raw numbers) — catching both
#: 'task-count snapshot' phrasings and the temporal_fact category reference.
#:
#: The numeric-signal branch (is_count_snapshot) handles memory_stale findings
#: that quote raw paired count strings; these markers handle the missing_knowledge
#: 'absence' shape that carries no numbers.
#:
#: NOTE: bare 'count snapshot' / 'count-snapshot' are intentionally excluded —
#: they are substrings of unrelated phrases such as 'account snapshot', which
#: could cause legitimate findings to be silently suppressed.  The more specific
#: 'task-count snapshot' and 'task count snapshot' already subsume all intended
#: phrasings produced by the Stage-3 LLM.
_TASK_COUNT_SNAPSHOT_MARKERS: tuple[str, ...] = (
    'task-count snapshot',
    'task count snapshot',
    'snapshot temporal_fact',
    'snapshot temporal fact',
    'task-count temporal',
)


def _is_task_count_snapshot_finding(flag: dict[str, Any]) -> bool:
    """Return True iff *flag* is about a task-count snapshot temporal_fact edge.

    Combines two detection strategies:
    1. Marker-phrase scan (catches missing_knowledge 'absence' findings that
       carry no raw count numbers): the combined description + suggested_action
       text contains any substring from :data:`_TASK_COUNT_SNAPSHOT_MARKERS`
       (case-insensitive).
    2. ``is_count_snapshot`` from task_filter (catches memory_stale findings
       that quote raw paired count strings like '607 done / 148 cancelled').

    Pure, sync, no I/O.
    """
    from fused_memory.reconciliation.task_filter import is_count_snapshot

    description = flag.get('description') or ''
    suggested_action = flag.get('suggested_action') or ''
    combined = f'{description} {suggested_action}'.lower()

    # Branch 1: marker-phrase scan
    if any(marker in combined for marker in _TASK_COUNT_SNAPSHOT_MARKERS):
        return True

    # Branch 2: raw count-string detection (handles numeric memory_stale shape)
    return is_count_snapshot(f'{description} {suggested_action}')


async def acknowledge_flag_marker(
    memory_service: Any,
    *,
    project_id: str,
    run_id: str,
    task_id: str,
    flag_type: str,
    mode: Literal['delete', 'tag'] = 'delete',
    log: logging.Logger = logger,
) -> int:
    """Acknowledge a prior ``stage1_flag_marker`` on the ledger.

    Looks up the marker row identified by ``(project_id, 'stage1_flag_marker',
    task_id, flag_type, run_id='')`` — the same identity :func:`dedup_flags`
    upserts to — and, if present, flips it to ``state='addressed'`` via
    ``memory_service.recon_ledger.mark_addressed``, stamping ``addressed_by``/
    ``addressed_run_id`` from *run_id* into its payload.

    ``mode`` (``'delete'`` or ``'tag'``) is accepted for call-site signature
    compatibility but both collapse to the same ``mark_addressed`` call — the
    ledger has no delete operation; ``state='addressed'`` IS the durable
    acknowledgement, so the Mem0-era delete-vs-tag distinction no longer
    applies.

    Guards: an invalid *task_id* (per :func:`_is_valid_marker_task_id`), an
    absent ``memory_service.recon_ledger`` (ledger disabled/not wired), or no
    matching row all short-circuit to a no-op ``0`` — the last case in
    particular means ``mark_addressed`` is never called against an identity
    with no row, so acknowledging an unknown/already-GC'd signature can never
    resurrect it. Never raises.

    ``addressed_by`` visibility to recurrence detection (amendment round 2,
    reviewer finding: design — confirmed safe by construction, unchanged by
    the ledger rewrite): :func:`dedup_flags` never drops/suppresses a flag on
    a HIT (it only annotates ``persisted_from_run``/``last_seen_run_id`` and
    re-upserts the marker row), so a genuine recurrence is still surfaced to
    Stage 2 even though the row was previously addressed; and the HIT-path
    upsert overwrites ``payload_json`` wholesale (no ``addressed_by`` key),
    so the acknowledgement self-clears on the very next recurrence rather than
    persisting indefinitely.  See
    ``test_dedup_flags_hit_on_addressed_marker_does_not_suppress_flag`` in
    ``test_flag_dedup.py`` for the pinned regression coverage.

    Args:
        memory_service: Service exposing ``.recon_ledger`` (a
            :class:`~fused_memory.reconciliation.recon_ledger.ReconLedgerStore`).
        project_id: Project scope for the ledger lookup/update.
        run_id: Current reconciliation run identifier; stamped as both
            ``addressed_by`` and ``addressed_run_id``.
        task_id: The flag's task_id (or comma-joined/``fp:`` signature key).
        flag_type: The flag's flag_type.
        mode: ``'delete'`` (default) or ``'tag'`` — accepted for compatibility,
            no longer changes behavior (see above).
        log: Logger for DEBUG messages; defaults to this module's logger.

    Returns:
        ``1`` if a matching row was found and acknowledged, else ``0``.
    """
    tid = str(task_id)
    ftype = str(flag_type)

    if not _is_valid_marker_task_id(tid):
        log.debug(
            'acknowledge_flag_marker: skipping invalid task_id %r flag_type %s',
            tid, ftype,
        )
        return 0

    ledger = getattr(memory_service, 'recon_ledger', None)
    if ledger is None:
        log.debug(
            'acknowledge_flag_marker: no recon_ledger on memory_service for task %s'
            ' flag_type %s (mode=%s) — no-op',
            tid, ftype, mode,
        )
        return 0

    probe = await ledger.get_by_identity(project_id, 'stage1_flag_marker', tid, ftype, '')
    if probe is None:
        return 0

    await ledger.mark_addressed(
        project_id,
        'stage1_flag_marker',
        tid,
        ftype,
        '',
        addressed_by=run_id,
        addressed_run_id=run_id,
    )
    return 1


async def acknowledge_resolved_flags(
    memory_service: Any,
    project_id: str,
    run_id: str,
    resolved_flags: list[dict[str, Any]],
    *,
    mode: Literal['delete', 'tag'] = 'delete',
    log: logging.Logger = logger,
) -> int:
    """Best-effort batch acknowledgment entry point for a list of resolved flags.

    For each flag in *resolved_flags*, computes its signature exactly as
    ``dedup_flags`` does — :func:`compute_flag_signature` first, falling back to
    :func:`compute_content_fingerprint_signature` — and delegates to
    :func:`acknowledge_flag_marker` for the ``(task_id, flag_type)`` pair.
    Flags for which both signature helpers return ``None`` are skipped (no I/O).

    Signatures are de-duplicated (order-preserving) BEFORE dispatch: two flags
    reducing to the same ``(task_id, flag_type)`` — e.g. two ``stale_metadata``
    findings on the same task — must acknowledge exactly once, not twice.
    Without this, two concurrent ``acknowledge_flag_marker`` calls for the same
    signature would each probe-then-``mark_addressed`` the SAME ledger row,
    inflating the returned count (amendment round 2, reviewer finding:
    robustness).

    Signable (de-duplicated) flags are dispatched to ``acknowledge_flag_marker``
    concurrently via :func:`~fused_memory.utils.async_utils.gather_collect`
    rather than one-at-a-time, since each is an independent ledger round-trip.

    Best-effort: a single flag's acknowledgment failing is logged at WARNING and
    does NOT abort the batch — the remaining flags are still processed. (This
    also guards against a caller-supplied replacement for
    ``acknowledge_flag_marker`` that raises, even though the real implementation
    never does.)

    Args:
        memory_service: Service exposing ``.recon_ledger``, forwarded to
            ``acknowledge_flag_marker``.
        project_id: Project scope forwarded to ``acknowledge_flag_marker``.
        run_id: Current reconciliation run identifier.
        resolved_flags: Flags whose requested action is already fulfilled.
        mode: ``'delete'`` (default) or ``'tag'``; forwarded verbatim to every
            ``acknowledge_flag_marker`` call in this batch.
        log: Logger for WARNING messages; defaults to this module's logger.

    Returns:
        Total count of markers acknowledged across all flags in the batch.
    """
    sigs: list[tuple[str, str]] = []
    for flag in resolved_flags:
        sig = compute_flag_signature(flag)
        if sig is None:
            sig = compute_content_fingerprint_signature(flag)
        if sig is None:
            continue
        sigs.append(sig)

    if not sigs:
        return 0

    # De-duplicate signatures before the fan-out (amendment round 2, reviewer
    # finding: robustness): dict.fromkeys preserves first-seen order while
    # dropping repeats, so a batch with two flags sharing one (task_id,
    # flag_type) signature acknowledges it exactly once instead of racing two
    # concurrent acknowledge_flag_marker calls against the same prior marker(s).
    deduped_sigs = list(dict.fromkeys(sigs))

    # Two-tier check via gather_collect (fused_memory.utils.async_utils).
    # Pass 1 (inside gather_collect): re-raises structured-cancellation
    # signals — this preserves the structured-cancellation contract and
    # prevents this batch fan-out from silently converting a shutdown
    # signal into an under-counted acknowledgment tally.
    # Pass 2 (below): per-item degrade-to-warning on ordinary Exceptions.
    results = await gather_collect(
        acknowledge_flag_marker(
            memory_service,
            project_id=project_id,
            run_id=run_id,
            task_id=tid,
            flag_type=ftype,
            mode=mode,
            log=log,
        )
        for tid, ftype in deduped_sigs
    )
    total = 0
    for (tid, ftype), result in zip(deduped_sigs, results, strict=True):
        if isinstance(result, Exception):
            log.warning(
                'acknowledge_resolved_flags: acknowledge_flag_marker failed for task %s'
                ' flag_type %s: %s',
                tid, ftype, result,
            )
        else:
            total += result
    return total


def filter_blocked_snapshot_findings(
    flags: list[dict[str, Any]],
    project_id: str,
) -> list[dict[str, Any]]:
    """Drop Stage-3 false-positive findings about blocked task-count snapshot edges.

    For projects in :data:`SNAPSHOT_WRITE_BLOCKED_PROJECTS`, the ABSENCE or
    staleness of a task-count snapshot temporal_fact edge is the CORRECT state
    (both write paths are blocked-by-design).  Stage 3 findings asserting the
    edge is missing or stale are false positives and must be suppressed.

    A flag is DROPPED iff **all three** conditions hold:
    1. ``project_id`` is in :data:`SNAPSHOT_WRITE_BLOCKED_PROJECTS`.
    2. ``flag['category']`` is in :data:`_SUPPRESSED_SNAPSHOT_CATEGORIES`
       (``missing_knowledge`` or ``memory_stale``).
    3. :func:`_is_task_count_snapshot_finding` returns ``True`` for the flag.

    All other flags pass through unchanged (fail-open).  The blast radius is
    tight: only registered projects × two categories × matching signature.

    A ``logger.debug`` line is emitted per dropped flag for observability.

    Args:
        flags: List of flag dicts from Stage 3 ``items_flagged``.
        project_id: The project being reconciled.

    Returns:
        Filtered list with false-positive blocked-snapshot findings removed.
    """
    from fused_memory.reconciliation.policies import is_snapshot_write_blocked

    if not is_snapshot_write_blocked(project_id):
        # Fail-open: project is not in the blocked set; return all flags unchanged.
        return list(flags)

    kept: list[dict[str, Any]] = []
    for flag in flags:
        category = flag.get('category') or ''
        if category in _SUPPRESSED_SNAPSHOT_CATEGORIES and _is_task_count_snapshot_finding(flag):
            logger.debug(
                'filter_blocked_snapshot_findings: dropping %s finding for task_id=%s '
                '(snapshot writes blocked-by-design for project %s)',
                category,
                flag.get('task_id'),
                project_id,
            )
            # do NOT append — flag is dropped
        else:
            kept.append(flag)

    return kept
