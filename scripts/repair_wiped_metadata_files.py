#!/usr/bin/env python3
"""Repair ``metadata.files`` wiped by the DONE path — the WRITE counterpart to
``scripts/audit_wiped_metadata_files.py``.

Task 3329. ``TaskWorkflow._reconcile_metadata_files_for_done``
(orchestrator/src/orchestrator/workflow.py:2007) blanks a task's
``metadata.files`` on two branches — reaching DONE with no merge sha, and its
``if err is not None: files = []`` arm, which fires WITH a real sha present.
Task 3113 owns the forward fix and is still ``pending``; this script backfills
the records the wiper already damaged, using the plan scope the audit recovers.

WHAT THIS SCRIPT REFUSES TO DO
------------------------------
It NEVER consumes a pasted or stale candidate payload. It re-runs
``audit_wiped_metadata_files.audit_project()`` IN-PROCESS on every invocation.
That is not fastidiousness — it is measured. Across this task's planning
sessions three read-only audit runs returned 40, then 43, then 45 candidates,
and in between one task (id 3086) MOVED from the repairable set into
``contradicted_real_merge_sha``. Repairing from the older payload would have
written files onto a task the contradicted rule forbids touching. The wiper is
live, so the population keeps drifting; only a fresh sweep is safe.

SAFETY MODEL
------------
Dry-run is the DEFAULT; ``--apply`` is required to write anything, and the MCP
client is constructed only on the apply path, so a bare invocation never even
dials the server. Four gates stand between a candidate and a write:

1. ``select_repairable_candidates`` — file-level fidelity only, never
   contradicted (see that function).
2. ``plan_files_rejection_reason`` — the lock-charter pre-check.
3. ``classify_live_task`` — a live ``get_task`` re-read immediately before the
   write; anything not terminal, and anything whose files are already present,
   is skipped.
4. ``update_task(metadata_mode='merge')`` — shallow merge, never ``replace``.
   Replace is the wiper's own primitive and a repair must not use it.

Usage:
    python scripts/repair_wiped_metadata_files.py --project-root <root>          # dry run
    python scripts/repair_wiped_metadata_files.py --project-root <root> --apply  # write
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from audit_wiped_metadata_files import (
    discover_project_roots,
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    AuditCoverage,
    WipeCandidate,
    _coerce_file_list,
    _format_coverage,
    audit_project,
)

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would silently evaluate the lock
# charter using the main checkout's predicate. Same reasoning and same form as
# scripts/reviewer_redundancy_diagnostic.py:35-38 (tasks 2881/2882).
_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.locking import directory_locks  # noqa: E402


def select_repairable_candidates(
    candidates: list[WipeCandidate],
) -> list[WipeCandidate]:
    """Return the subset of *candidates* this repair may write to.

    Two constraints, both expressed as EXCLUSIONS rather than as an inclusion
    list of known-good signatures:

    * ``plan_files_fidelity`` must be :data:`FIDELITY_FILE_LEVEL`. A
      lock-level record holds MODULE paths, not plan.files entries; writing
      one into ``metadata.files`` would corrupt the very record it claims to
      repair. Checked independently of the signature — a confirmed wipe
      signature does not license writing a module path.
    * ``wipe_signature`` must not be :data:`CONTRADICTED_REAL_MERGE_SHA`. Such
      a task has BOTH a null-sha row and a real merge sha, i.e. a failed merge
      attempt that was retried and landed, so the null-sha DONE path is not
      what emptied it.

    WHY EXCLUSION AND NOT INCLUSION. An inclusion list of the three signatures
    the task description enumerates — ``confirmed_null_sha_done_path``,
    ``no_successful_merge_sha``, ``no_merge_event`` — would silently drop
    :data:`audit_wiped_metadata_files.CLEAN_MERGE_SHA`, a class that did not
    exist in that snapshot but has members today. The audit's own constant
    docstring states clean_merge_sha is explicitly NOT an exoneration, so
    dropping it would have left real damage unrepaired and unreported. An
    exclusion rule fails OPEN toward review: a class the audit grows later
    reaches the report and the operator, instead of vanishing from the feed.

    Input order is preserved (it is the operator's audit trail) and the input
    list is never mutated.
    """
    return [
        candidate
        for candidate in candidates
        if candidate.plan_files_fidelity == FIDELITY_FILE_LEVEL
        and candidate.wipe_signature != CONTRADICTED_REAL_MERGE_SHA
    ]


def plan_files_rejection_reason(candidate: WipeCandidate) -> str | None:
    """Return why *candidate*'s plan_files cannot be written, or None if they can.

    THE LOCK-CHARTER PRE-CHECK. ``_reject_directory_locks_in_update_metadata``
    (fused-memory/src/fused_memory/middleware/task_interceptor.py:5284) rejects
    any ``update_task`` metadata write whose ``files`` list carries a
    DIRECTORY-classified entry, raising a ``lock_charter_error``. Discovering
    that mid-batch gives an operator an opaque MCP failure on task N of 35;
    pre-checking converts it into a named, attributable skip that says which
    path is at fault, in the same summary as every other disposition.

    The predicate is :func:`shared.locking.directory_locks`, IMPORTED, never
    re-implemented. ``shared/src/shared/locking.py``'s own docstring records
    that a drifting copy undercounting the extension allowlist by 22 entries is
    exactly what incident #3117 was; ``fused_memory.middleware.lock_charter_guard``
    duplicates it verbatim for the same reason. Note this correctly accepts
    systemd ``.timer``/``.service`` units, which are in ``CODE_EXTENSIONS``.

    An EMPTY (or all-blank) plan_files is rejected separately and first: there
    is nothing to restore, and a write setting ``files`` to ``[]`` would
    re-perform the very wipe this script exists to undo.

    Measured against today's population: 0 of the 140 repairable path entries
    trip the directory arm. This gate exists because the candidate population
    drifts under a live wiper, not because it currently fires.
    """
    files = [f for f in candidate.plan_files if isinstance(f, str) and f.strip()]
    if not files:
        return (
            "recovered plan_files is empty — nothing to restore (a write "
            "setting files to [] would repeat the wipe, so none is issued)"
        )

    offenders = directory_locks(list(files))
    if offenders:
        return (
            "plan_files carries directory-classified entries the interceptor's "
            "lock charter would reject (lock_charter_error): "
            + ", ".join(offenders)
        )
    return None


# ---------------------------------------------------------------------------
# The write payload.
# ---------------------------------------------------------------------------

# This task's id, stamped into every backfill so a later reader can attribute
# the write without consulting the event log.
REPAIR_TASK_ID = "3329"

# WHY x_-PREFIXED, and not the bare `files_backfill_provenance` the task
# description names. docs/task-authoring.md §Tier-C is explicit that a one-off
# annotation key "must never be filed as a bespoke top-level metadata key —
# that just adds another code=unknown_key census line. Use the x_-prefixed
# forward-compat namespace instead — silently allowed, no warning."  Verified
# in code: shared/src/shared/task_metadata.py:933 exempts x_-prefixed keys from
# the unknown_key warning, and `files_backfill_provenance` is NOT in
# _BLESSED_METADATA_KEYS. A bare key would therefore emit one
# `task_metadata.schema_warning code=unknown_key` line per repaired task — 35+
# lines of precisely the drift signal that census exists to surface, spent on a
# one-off repair annotation. The record's SEMANTIC content (this task's id, the
# recovery source, the wipe signature) is unchanged from the description's
# spec; only the key spelling honours the repo's documented vocabulary. Grep
# found zero in-repo consumers of either spelling.
PROVENANCE_KEY = "x_files_backfill_provenance"


def build_repair_payload(candidate: WipeCandidate, *, now_iso: str) -> dict:
    """Build the metadata patch that restores *candidate*'s wiped file scope.

    A MINIMAL ADDITIVE PATCH: exactly two keys, both of which this write is
    responsible for. That minimality is load-bearing three times over.

    * No ``status``. ``SqliteTaskBackend.update_task`` raises
      ``StatusWriteAuthorityError`` (backends/sqlite_task_backend.py:2575) for a
      status carried on a metadata write — ``set_task_status`` is the only
      sanctioned writer.
    * No ``done_provenance``, at any nesting depth, for the same reason via
      ``DoneProvenanceWriteAuthorityError``.
    * No ``modules`` and no ``files_tagged_at``. This restores a scope record;
      it does not re-run the module tagger, and claiming a fresh tagging would
      be a false provenance stamp.

    Because the write goes out under ``metadata_mode='merge'``, every key NOT
    named here is left untouched on the stored blob — which is the whole point,
    since the wipe class being repaired is precisely a whole-blob overwrite.

    ``now_iso`` is INJECTED rather than stamped from the clock inside, so the
    caller decides the batch timestamp (one value for a whole run) and tests
    can assert an exact string.
    """
    return {
        "files": list(candidate.plan_files),
        PROVENANCE_KEY: {
            "task": REPAIR_TASK_ID,
            "src": candidate.plan_files_source,
            "sig": candidate.wipe_signature,
            "fidelity": candidate.plan_files_fidelity,
            "at": now_iso,
        },
    }


# ---------------------------------------------------------------------------
# Dispositions.
#
# EVERY candidate the audit surfaces ends up under exactly one of these, and
# every one of them is printed with its count — including the zero ones. A
# disposition that is merely absent from a report reads as "did not happen"
# when it may well mean "was never evaluated".
# ---------------------------------------------------------------------------

# Annotated `: str` rather than left to inference: a bare assignment infers as
# Literal['repair'] etc., which makes dict.fromkeys(ALL_DISPOSITIONS, 0) a
# dict[Literal[...], int] that no longer satisfies a dict[str, int] signature.
# These are an open vocabulary of report keys, not a closed enum.
REPAIR: str = "repair"                           # written (or would be, in a dry run)
SKIP_CONTRADICTED: str = "skip_contradicted"     # constraint 2, dropped by selection
SKIP_LOCK_LEVEL_FIDELITY: str = "skip_lock_level_fidelity"  # constraint 1, dropped by selection
SKIP_LOCK_CHARTER: str = "skip_lock_charter"     # the interceptor would reject the files
SKIP_NOT_TERMINAL: str = "skip_not_terminal"     # live task is not done/cancelled
SKIP_FILES_PRESENT: str = "skip_files_present"   # already has a scope; re-run safe
SKIP_MISSING: str = "skip_missing"               # live re-read returned nothing usable
FAILED: str = "failed"                           # the write was attempted and errored

# Report order, and the exhaustive list the summary iterates. Printed in full
# on EVERY run, zero counts included.
ALL_DISPOSITIONS: tuple[str, ...] = (
    REPAIR,
    SKIP_CONTRADICTED,
    SKIP_LOCK_LEVEL_FIDELITY,
    SKIP_LOCK_CHARTER,
    SKIP_NOT_TERMINAL,
    SKIP_FILES_PRESENT,
    SKIP_MISSING,
    FAILED,
)

# Dispositions whose entries are listed INDIVIDUALLY, with id and reason. An
# aggregate "2 failed" tells an operator nothing they can act on.
_ITEMISED_DISPOSITIONS: tuple[str, ...] = (
    FAILED,
    SKIP_LOCK_CHARTER,
    SKIP_CONTRADICTED,
    SKIP_LOCK_LEVEL_FIDELITY,
    SKIP_NOT_TERMINAL,
    SKIP_MISSING,
)

# Statuses this repair may write to. An ALLOWLIST, deliberately, not a
# "not in {pending, in-progress, ...}" denylist: a status the system grows
# later must fail CLOSED (skipped and reported) rather than silently becoming
# writable underneath a workflow nobody has considered yet.
TERMINAL_STATUSES = frozenset({"done", "cancelled"})


def classify_live_task(live_task: object, candidate: WipeCandidate) -> str:
    """Decide what to do with *candidate* given its LIVE ``get_task`` re-read.

    Called immediately before each write, never resolved in advance. Task
    3113's correction addendum documents ``_stamp_optimistic_path``
    (orchestrator/src/orchestrator/workflow.py:4550 — located by NAME; it moved
    from 4413 while this plan was open) writing a task's stale dispatch-time
    ``metadata`` snapshot back as a whole blob, which is the DF 3260 clobber
    class. Repairing underneath a live workflow is therefore not merely racy,
    it is guaranteed to be undone — hence :data:`SKIP_NOT_TERMINAL`.

    :data:`SKIP_FILES_PRESENT` is what makes this script idempotent: a second
    pass (the one the addendum predicts will be needed once 3113 lands) re-runs
    cheaply and touches only what is still empty.

    The emptiness test reuses the audit's ``_coerce_file_list`` rather than
    bare truthiness, so the predicate deciding "this task still has no scope"
    is byte-identical to the one that nominated the candidate. A divergent one
    would let the audit and the repair disagree about the same record — the
    audit reporting damage the repair silently declines to fix.
    """
    if not isinstance(live_task, dict) or not live_task.get("status"):
        return SKIP_MISSING
    if str(live_task.get("status")) not in TERMINAL_STATUSES:
        return SKIP_NOT_TERMINAL

    metadata = live_task.get("metadata")
    current_files = (
        _coerce_file_list(metadata.get("files")) if isinstance(metadata, dict) else ()
    )
    if current_files:
        return SKIP_FILES_PRESENT
    return REPAIR


# ---------------------------------------------------------------------------
# The write path.
# ---------------------------------------------------------------------------


class _ToolClient(Protocol):
    """The one method this script needs from an MCP client (fake or real)."""

    async def call_tool(self, name: str, arguments: dict) -> dict: ...


class RepairOutcome(NamedTuple):
    """What happened to ONE candidate, and why.

    ``detail`` carries the operator-actionable reason for every non-REPAIR
    disposition (the rejecting path, the error text) so the summary never has
    to say "skipped" without saying why.
    """

    task_id: int
    tag: str
    disposition: str
    files: tuple[str, ...]
    detail: str | None = None


def _error_detail(result: object) -> str | None:
    """Return an error description if *result* is an error-shaped reply, else None.

    The server can report a rejection by ANSWERING rather than raising — an
    ``{"error": ...}`` body, or a ``{"success": False}`` flag. A bare truthiness
    check on the reply would count both as a repair that never happened, which
    is the silent fail-soft this script exists to avoid. Anything else
    (including an empty dict, or a plain echoed task) is a success: the
    ABSENCE of an error marker is not itself an error marker.
    """
    if not isinstance(result, dict):
        return None
    if result.get("error"):
        return f"server returned an error: {result['error']}"
    if result.get("success") is False:
        detail = result.get("error_type") or result.get("message") or result
        return f"server reported failure: {detail}"
    return None


async def repair_one(
    client: _ToolClient,
    project_root: str,
    candidate: WipeCandidate,
    *,
    now_iso: str,
) -> RepairOutcome:
    """Write ONE candidate's recovered scope back, and never raise.

    Mode is ``metadata_mode='merge'`` — a SHALLOW merge that leaves every key
    this payload does not name untouched. Emphatically NOT ``'replace'``: that
    is the exact primitive behind the wipe class being repaired
    (``_execute_combine``, task_interceptor.py:1850, writes a 3-key blob in
    replace mode; ``_merge_metadata``, sqlite_task_backend.py:3301, returns
    ``incoming`` verbatim for that mode, deleting ``files``, ``spawned_from``,
    ``source``, ``branch_base_sha`` and the rest). A repair must not use the
    wiper's own primitive. Merge additionally raises loudly on a corrupt stored
    blob instead of clobbering it.

    ERRORS ARE RETURNED, NOT RAISED. ``candidate_key`` is recomputed from
    (title, files) on every metadata-touching update, so backfilling ``files``
    onto a batch can legitimately collide and raise
    ``DuplicateCandidateKeyError``. One collision on task 12 of 35 must not
    strand the remaining 23: the failure is captured, attributed and reported,
    and drives a non-zero exit at the end.
    """
    payload = build_repair_payload(candidate, now_iso=now_iso)
    try:
        result = await client.call_tool(
            "update_task",
            {
                # No `status` and no done_provenance anywhere in the payload —
                # the StatusWriteAuthorityError / DoneProvenanceWriteAuthorityError
                # floors (sqlite_task_backend.py:2575+).
                "id": str(candidate.task_id),
                # LOAD-BEARING, and passed VERBATIM — never coalesced to
                # 'master', never made conditional on being non-default. `tag`
                # is half the `(tag, id)` primary key the audit nominates
                # candidates under, and the audit read it straight out of the
                # tasks.db `tag` column, so it is always a real tag. Omitting it
                # would not raise: the MCP layer substitutes
                # DEFAULT_TAG = 'master' (sqlite_task_backend.py:127, applied on
                # the update_task path at :2612), silently writing the recovered
                # scope onto a DIFFERENT task that merely shares the id. Any
                # coalescing here just re-introduces that defaulting bug one
                # layer up.
                "tag": candidate.tag,
                "project_root": project_root,
                "metadata": json.dumps(payload),
                "metadata_mode": "merge",
            },
        )
    except Exception as exc:  # noqa: BLE001 — a per-task failure, never fatal
        return RepairOutcome(
            task_id=candidate.task_id,
            tag=candidate.tag,
            disposition=FAILED,
            files=candidate.plan_files,
            detail=f"{type(exc).__name__}: {exc}",
        )

    detail = _error_detail(result)
    if detail is not None:
        return RepairOutcome(
            task_id=candidate.task_id,
            tag=candidate.tag,
            disposition=FAILED,
            files=candidate.plan_files,
            detail=detail,
        )
    return RepairOutcome(
        task_id=candidate.task_id,
        tag=candidate.tag,
        disposition=REPAIR,
        files=candidate.plan_files,
    )


class RepairResult(NamedTuple):
    """One project's repair run: every disposition, plus the audit's coverage.

    ``coverage`` travels with the outcomes VERBATIM, never recomputed or
    summarised, because the candidate list is an observable subset and a reader
    who sees only the outcomes cannot tell a clean project from an unobservable
    one.
    """

    project_root: str
    applied: bool
    outcomes: list[RepairOutcome]
    coverage: AuditCoverage


async def repair_project(
    client: _ToolClient | None,
    project_root: str,
    *,
    apply: bool,
    now_iso: str,
) -> RepairResult:
    """Audit *project_root* fresh, then repair what is safe to repair.

    Sequence, in order:

    1. ``audit_project`` IN-PROCESS — a fresh sweep every run. Never a pasted
       payload; see the module docstring for the measured drift that makes a
       stale feed unsafe.
    2. ``select_repairable_candidates``. The dropped candidates are RECORDED
       under :data:`SKIP_CONTRADICTED` / :data:`SKIP_LOCK_LEVEL_FIDELITY`
       rather than silently filtered away, so an exclusion is visible in the
       report as a decision rather than as an absence.
    3. ``plan_files_rejection_reason`` — the lock-charter pre-check.
    4. A live ``get_task`` re-read per surviving candidate, then
       ``classify_live_task``.
    5. On :data:`REPAIR` AND only when *apply* is true, ``repair_one``.

    Writes are SEQUENTIAL, not gathered. The interceptor serialises per-project
    writes under its own ``_write_lock`` anyway, so concurrency would buy
    nothing, and a sequential loop keeps the per-task disposition log ordered
    and attributable.

    *client* may be None on the dry-run path — that is what lets a dry run
    avoid dialling the server at all. A None client with ``apply=True`` is a
    programming error and raises.
    """
    if apply and client is None:
        raise ValueError("apply=True requires an MCP client")

    audit = audit_project(project_root)
    repairable = select_repairable_candidates(audit.candidates)
    repairable_ids = {(c.tag, c.task_id) for c in repairable}

    outcomes: list[RepairOutcome] = []

    # (2) Record the exclusions explicitly.
    for candidate in audit.candidates:
        if (candidate.tag, candidate.task_id) in repairable_ids:
            continue
        if candidate.wipe_signature == CONTRADICTED_REAL_MERGE_SHA:
            disposition = SKIP_CONTRADICTED
            detail = (
                "a null-sha row and a REAL merge sha coexist (a failed merge "
                "attempt that was retried and landed), so the null-sha DONE "
                "path is not what emptied this task"
            )
        else:
            disposition = SKIP_LOCK_LEVEL_FIDELITY
            detail = (
                f"recovered scope is {candidate.plan_files_fidelity}, not "
                f"{FIDELITY_FILE_LEVEL}: these are MODULE paths, not plan.files "
                "entries, and must never be written into metadata.files as-is"
            )
        outcomes.append(
            RepairOutcome(
                task_id=candidate.task_id,
                tag=candidate.tag,
                disposition=disposition,
                files=candidate.plan_files,
                detail=detail,
            )
        )

    for candidate in repairable:
        # (3) Lock-charter pre-check, before any network call.
        reason = plan_files_rejection_reason(candidate)
        if reason is not None:
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=SKIP_LOCK_CHARTER,
                    files=candidate.plan_files,
                    detail=reason,
                )
            )
            continue

        # (4) Live re-read. On the dry-run path there is no client, so the
        # live state is unknown and the candidate is reported as a would-be
        # repair — the dry run's job is to show the SELECTION, and the gate
        # is re-evaluated for real at write time.
        if client is None:
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=REPAIR,
                    files=candidate.plan_files,
                    detail=(
                        "dry run: live status not re-read (no MCP call made); "
                        "the terminal/files-present gate is evaluated at write time"
                    ),
                )
            )
            continue

        try:
            live = await client.call_tool(
                "get_task",
                {
                    "id": str(candidate.task_id),
                    # VERBATIM, for the same reason as the update_task call in
                    # repair_one: `tag` is half the `(tag, id)` key this
                    # candidate was nominated under, and get_task defaults a
                    # missing tag to DEFAULT_TAG = 'master' rather than erroring.
                    # Dropped here, THIS gate judges the wrong task's live status
                    # — and idempotency dies with it, because SKIP_FILES_PRESENT
                    # would be evaluated against the master-tag row, so a re-run
                    # would re-write.
                    "tag": candidate.tag,
                    "project_root": project_root,
                },
            )
        except Exception as exc:  # noqa: BLE001 — one unreadable task, not a batch abort
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=SKIP_MISSING,
                    files=candidate.plan_files,
                    detail=f"live re-read failed: {type(exc).__name__}: {exc}",
                )
            )
            continue

        disposition = classify_live_task(_unwrap_task(live), candidate)
        if disposition != REPAIR:
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=disposition,
                    files=candidate.plan_files,
                    detail="live re-read immediately before the write",
                )
            )
            continue

        # (5) Write.
        outcomes.append(await repair_one(client, project_root, candidate, now_iso=now_iso))

    return RepairResult(
        project_root=project_root,
        applied=apply,
        outcomes=outcomes,
        coverage=audit.coverage,
    )


def _unwrap_task(payload: Any) -> Any:
    """Return the task dict from a ``get_task`` reply.

    The MCP reply may be the task itself or may wrap it under a ``task`` key.
    Anything else is passed through untouched, so ``classify_live_task``
    (which fails closed on an unrecognised shape) makes the call rather than
    this helper guessing.
    """
    if isinstance(payload, dict) and isinstance(payload.get("task"), dict):
        return payload["task"]
    return payload


# ---------------------------------------------------------------------------
# Reporting — the honesty artifact.
# ---------------------------------------------------------------------------

# Printed on EVERY run, including one that repaired nothing. The candidate list
# is an observable subset; the unobservable remainder is not "clean", it is
# UNKNOWN. Omitting this on a quiet run is precisely the no-silent-fail-soft
# violation named in docs/legibility/design-invariants.md, and it is what would
# let this repair be reported as "fixed the blast radius" when it fixed only
# the part that could be seen.
_OBSERVABLE_SUBSET_CAVEAT = (
    "  SCOPE OF THIS RUN: the candidates above are an OBSERVABLE SUBSET, not "
    "the full",
    "  damaged population. Tasks with no recoverable plan scope are UNKNOWN — "
    "neither",
    "  clean nor damaged — and are NOT repaired here. Do not report this run as "
    '"fixed',
    '  the blast radius"; it fixed the observable part of it.',
)

_DRY_RUN_BANNER = (
    "  DRY RUN — NO WRITE WAS ATTEMPTED. Candidates listed under "
    f"'{REPAIR}' are what",
    "  an --apply run would write; their live status is re-read and re-judged "
    "at that",
    "  point, so this list is a selection, not a promise.",
)


def _counts(result: "RepairResult") -> dict[str, int]:
    """Count EVERY disposition, seeding the zero ones explicitly."""
    counts = dict.fromkeys(ALL_DISPOSITIONS, 0)
    for outcome in result.outcomes:
        counts[outcome.disposition] = counts.get(outcome.disposition, 0) + 1
    return counts


def format_summary(result: "RepairResult") -> str:
    """Render one project's repair run as a human-readable summary.

    Three properties this owes its reader, each of them tested:

    * Every disposition bucket is printed WITH ITS COUNT, zeros included. A
      bucket that is merely absent reads as "did not happen" when it may mean
      "was never evaluated".
    * Every failure and every skip that an operator could act on is listed
      INDIVIDUALLY, with its task id and its reason.
    * The audit's coverage block and the observable-subset caveat are printed
      unconditionally, including on a run that repaired nothing.
    """
    counts = _counts(result)
    mode = "APPLY" if result.applied else "DRY RUN"
    lines = [f"{result.project_root}: [{mode}]"]

    if not result.applied:
        lines.extend(_DRY_RUN_BANNER)

    lines.append("  -- dispositions --")
    for disposition in ALL_DISPOSITIONS:
        lines.append(f"    {disposition:<26} {counts[disposition]}")

    for disposition in _ITEMISED_DISPOSITIONS:
        entries = [o for o in result.outcomes if o.disposition == disposition]
        if not entries:
            continue
        lines.append(f"  -- {disposition} ({len(entries)}) --")
        for outcome in entries:
            lines.append(
                f"    task_id={outcome.task_id} tag={outcome.tag}: "
                f"{outcome.detail or 'no detail recorded'}"
            )

    repaired = [o for o in result.outcomes if o.disposition == REPAIR]
    if repaired:
        verb = "repaired" if result.applied else "would repair"
        lines.append(f"  -- {verb} ({len(repaired)}) --")
        for outcome in repaired:
            lines.append(
                f"    task_id={outcome.task_id} tag={outcome.tag} "
                f"files={len(outcome.files)}"
            )

    # Reuse the audit's own coverage phrasing rather than paraphrasing it —
    # one wording for one claim, so the two reports can never drift into
    # describing the same counts differently.
    lines.extend(_format_coverage(result.coverage))
    lines.extend(_OBSERVABLE_SUBSET_CAVEAT)
    return "\n".join(lines)


def format_json_summary(results: list["RepairResult"]) -> str:
    """Render *results* as a JSON OBJECT carrying dispositions AND coverage.

    An object rather than a bare array, mirroring
    ``audit_wiped_metadata_files.format_json``: the coverage counts must travel
    with the outcomes, or a consumer that received only a list of repairs could
    not tell a clean project from an unobservable one.
    """
    return json.dumps(
        {
            "projects": [
                {
                    "project_root": result.project_root,
                    "applied": result.applied,
                    "coverage": result.coverage._asdict(),
                    "counts": _counts(result),
                    "outcomes": [
                        {**o._asdict(), "files": list(o.files)}
                        for o in result.outcomes
                    ],
                    "observable_subset_caveat": " ".join(
                        line.strip() for line in _OBSERVABLE_SUBSET_CAVEAT
                    ),
                }
                for result in results
            ]
        }
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

DEFAULT_SERVER = "http://127.0.0.1:8002"

# WHY THIS NAME IS LOAD-BEARING, not cosmetic. fused_memory/server/tools.py:524
# derives a write's `agent_id` from ctx.session.client_params.clientInfo.name.
# Inheriting the migrate script's 'migrate-metadata' would file every one of
# these repair writes in the journal and the event stream under an unrelated,
# long-finished migration — making the backfill unattributable exactly when
# someone is trying to work out who touched a historical record.
CLIENT_NAME = "repair-wiped-metadata-files-3329"


def _make_client(server_url: str):
    """Construct the MCP client. Imported LAZILY, on the apply path only.

    ``migrate_metadata_modules_to_files`` imports httpx at module scope, so
    importing it at the top of this module would make a dry run depend on a
    transport library it never uses. Subclassed rather than edited: that file
    is outside this task's lock scope, and the migrate script's update_task
    recipe is the proven one — only the mode differs (merge, not replace).
    """
    from migrate_metadata_modules_to_files import FusedMemoryClient

    class RepairFusedMemoryClient(FusedMemoryClient):
        """FusedMemoryClient with an attributable clientInfo.name."""

        async def _initialize(self) -> None:
            await self._post({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": CLIENT_NAME, "version": "1.0"},
                    "capabilities": {},
                },
            })
            await self._post({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            })

    return RepairFusedMemoryClient(server_url)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair metadata.files wiped by the DONE path. DRY RUN BY DEFAULT: "
            "without --apply nothing is written and the MCP server is never "
            "even dialled. Re-runs the read-only audit in-process on every "
            "invocation -- it never consumes a stale candidate list."
        ),
        epilog=(
            "exit codes: 0 = ran, nothing failed; 1 = at least one candidate "
            "FAILED to write; 2 = no project root resolved to a readable "
            "tasks.db (nothing was examined -- never read 2 as a clean run)."
        ),
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to repair (resolves <root>/.taskmaster/tasks/tasks.db, "
            "<root>/data/orchestrator/runs.db and <root>/.worktrees). "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--apply", action="store_true",
        help=(
            "Actually write. Without this the run is inert: no MCP client is "
            "constructed and no task is touched."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object (dispositions + coverage) instead of a report.",
    )
    parser.add_argument(
        "--server-url", default=DEFAULT_SERVER,
        help=f"Fused-memory MCP server URL (default: {DEFAULT_SERVER}).",
    )
    return parser


async def main_async(args: argparse.Namespace) -> int:
    roots = discover_project_roots(project_roots=args.project_roots)
    if not roots:
        print(
            "no project root resolvable with a readable tasks.db (checked "
            "--project-root / DASHBOARD_KNOWN_PROJECT_ROOTS / the "
            "dark-factory default); NOTHING was examined",
            file=sys.stderr,
        )
        return 2

    now_iso = datetime.now(timezone.utc).isoformat()
    results: list[RepairResult] = []

    if args.apply:
        # The client is constructed ONLY here, so a dry run genuinely never
        # dials — that inertness is what makes an accidental bare invocation
        # safe, and it is asserted by pointing --server-url at a closed port.
        async with _make_client(args.server_url) as client:
            for root in roots:
                results.append(
                    await repair_project(client, root, apply=True, now_iso=now_iso)
                )
    else:
        for root in roots:
            results.append(
                await repair_project(None, root, apply=False, now_iso=now_iso)
            )

    if args.json:
        print(format_json_summary(results))
    else:
        print("\n".join(format_summary(result) for result in results))

    failed = sum(
        1
        for result in results
        for outcome in result.outcomes
        if outcome.disposition == FAILED
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = ran, nothing failed; 1 = at least one candidate FAILED to
    write; 2 = no project root resolved to a readable tasks.db.

    2 is distinct from 0 for the same reason the audit distinguishes them: 0
    would otherwise be returned both for "examined everything, nothing to do"
    and for "examined nothing at all", and a consumer reading only the exit
    code would take a total no-op for a clean run.
    """
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except sqlite3.Error as exc:
        print(f"error: could not read a project database: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
