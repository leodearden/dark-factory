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

Gate 3's live re-read now fails LOUDLY. A re-read the server never usefully
answered — a dead transport, or an empty/unreadable reply, both classified by
``classify_reply`` — is reported as :data:`FAILED_LIVE_READ` and drives a
non-zero exit (:data:`EXIT_LIVE_READ_FAILED`), instead of being filed as a
benign :data:`SKIP_MISSING` indistinguishable from a task the server
genuinely said is gone. This closes the "report a repair that never
happened" gap for the READ path — the same gap ``classify_reply`` already
closes for the WRITE path in ``repair_one``, where a write this script cannot
confirm succeeded is reported as :data:`FAILED` rather than as a silent
:data:`REPAIR`.

Usage:
    python scripts/repair_wiped_metadata_files.py --project-root <root>          # dry run
    python scripts/repair_wiped_metadata_files.py --project-root <root> --apply  # write
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from audit_wiped_metadata_files import (
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    AuditCoverage,
    WipeCandidate,
    _coerce_file_list,
    _format_coverage,
    audit_project,
    discover_project_roots,
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


def writable_plan_files(candidate: WipeCandidate) -> tuple[str, ...]:
    """Return EXACTLY the ``files`` entries a repair write may carry.

    ONE sanitiser, read by both the gate (:func:`plan_files_rejection_reason`)
    and the payload (:func:`build_repair_payload`), because a gate that
    validates a different list from the one that ships is not a gate. Without
    it a candidate whose plan_files is ``("a.py", "   ")`` passes the charter
    check on the filtered ``["a.py"]`` and then writes ``["a.py", "   "]``.

    The audit's ``_coerce_file_list`` deliberately KEEPS a whitespace-only
    entry (``"  "`` is a truthy ``str``) and it is right to: for the audit's
    purpose that entry is evidence the task HAD a scope. For a WRITE it is
    junk, and nothing downstream would catch it —
    ``shared.locking.directory_locks`` skips blanks, so the interceptor accepts
    it and it lands in ``metadata.files`` verbatim. The divergence is resolved
    here, in the one place both callers read, rather than by making the two
    predicates agree by inspection.
    """
    return tuple(f for f in candidate.plan_files if isinstance(f, str) and f.strip())


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
    systemd ``.timer``/``.service`` units, which are in ``FILE_EXTENSIONS``.

    An EMPTY (or all-blank) plan_files is rejected separately and first: there
    is nothing to restore, and a write setting ``files`` to ``[]`` would
    re-perform the very wipe this script exists to undo.

    Measured against today's population: 0 of the 140 repairable path entries
    trip the directory arm. This gate exists because the candidate population
    drifts under a live wiper, not because it currently fires.

    Judged on :func:`writable_plan_files` — the SAME list ``build_repair_payload``
    ships, so this validates the exact bytes that get written.
    """
    files = writable_plan_files(candidate)
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

    ``files`` is :func:`writable_plan_files`, NOT ``candidate.plan_files``. The
    two differ on blank/whitespace-only entries, and shipping the raw tuple
    while the lock-charter gate judged the filtered one would mean validating
    different bytes from the ones written — see that function.
    """
    return {
        "files": list(writable_plan_files(candidate)),
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
SKIP_MISSING: str = "skip_missing"               # the server ANSWERED and said the task is not there / not usable
FAILED_LIVE_READ: str = "failed_live_read"       # the live re-read never answered; candidate NEVER JUDGED
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
    FAILED_LIVE_READ,
    FAILED,
)

# Dispositions whose entries are listed INDIVIDUALLY, with id and reason. An
# aggregate "2 failed" tells an operator nothing they can act on.
_ITEMISED_DISPOSITIONS: tuple[str, ...] = (
    FAILED,
    FAILED_LIVE_READ,
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


class ReplyVerdict(NamedTuple):
    """The result of classifying one MCP server reply.

    ``ok`` mirrors ``fused_memory.middleware.task_interceptor.
    interceptor_write_succeeded`` (fused-memory/src/fused_memory/middleware/
    task_interceptor.py:5829-5868) verbatim in semantics — its four-clause
    rule, ``isinstance(resp, dict) and bool(resp) and
    bool(resp.get('success', True)) and not resp.get('error')`` — because
    that predicate is this repo's canonical write-success contract, and a
    reply this script cannot confirm succeeded IS a failure. That docstring's
    own words: "An empty dict {} carries no positive write signal and is
    therefore treated as **failure**"; "Non-dict responses (None, strings,
    lists) are always treated as failures." This is a TRANSCRIBED MIRROR, not
    an import — see :func:`classify_reply` for why importing the canonical
    predicate is infeasible under this suite's verify gate.

    ``answered`` is the axis ``interceptor_write_succeeded`` has no need of:
    whether the server explained ITSELF (an explicit ``error`` /
    ``success: False`` marker) or said nothing usable at all (``{}``, or any
    non-dict reply). :func:`repair_project`'s live re-read uses it to tell
    "the server told us this task is gone" (a benign, named skip) apart from
    "the server never answered usefully" (indistinguishable from a dead
    transport, and therefore a candidate that was never judged).

    ``detail`` is the operator-actionable reason for a non-``ok`` verdict,
    and is ``None`` iff ``ok`` is ``True``.
    """

    ok: bool
    answered: bool
    detail: str | None


def classify_reply(result: object) -> ReplyVerdict:
    """Classify one MCP server reply: did it succeed, did the server answer, why not.

    The server ALWAYS reports a rejection by ANSWERING rather than raising:
    ``@mcp_tool_errors`` (fused-memory/src/fused_memory/server/tool_errors.py:44-59)
    converts every exception into an ordinary reply dict, so nothing
    server-side ever crosses the wire as an exception. But the converse — a
    reply this script cannot read as a positive success signal — is not
    innocent either. This repo's canonical write-success contract,
    ``interceptor_write_succeeded`` (fused-memory/src/fused_memory/middleware/
    task_interceptor.py:5829-5868), treats an empty dict and any non-dict
    response as **failure**. THIS REPLACES THIS FUNCTION'S OWN FORMER STANCE
    — "the ABSENCE of an error marker is not itself an error marker" — which
    was the bug: a wedged server that ACKs with a 202 or an empty body
    reaches this script as literally ``{}``. See ``FusedMemoryClient._post``
    (scripts/migrate_metadata_modules_to_files.py:89-90), which returns
    ``{}`` for a 202 or an empty body, and ``call_tool`` (:126), which falls
    through to ``result.get('result', {})`` when the reply carries neither
    ``structuredContent`` nor text content — so ``{}`` is a genuinely
    reachable shape through this script's own transport, not a hypothetical.
    The old rule reported that reply, indistinguishable from a dead
    transport, as a REPAIR that never happened.

    ``answered`` exists for :func:`repair_project`'s live re-read, which must
    tell a benign "the server explained itself" skip apart from "the server
    never answered usefully at all" — see that function's docstring.

    THE CHECK ORDER IS LOAD-BEARING: not-a-dict first (nothing else can call
    ``.get`` on it), then emptiness (a ``{}`` has neither an ``error`` nor a
    ``success`` key and would otherwise fall through to the ok branch), and
    only then the error/success probes.

    ``error_type`` is included in the detail when present — it is the
    machine-readable half of the reply (``TaskNotFoundError``,
    ``DuplicateCandidateKeyError``) and is what an operator greps for.
    """
    if not isinstance(result, dict):
        raw = repr(result)
        if len(raw) > 200:
            raw = raw[:200] + "…"
        return ReplyVerdict(
            False, False, f"server reply was not a dict: {type(result).__name__}: {raw}"
        )
    if not result:
        return ReplyVerdict(
            False,
            False,
            "server reply was empty ({}) — it carries no positive write signal",
        )
    if result.get("error"):
        error_type = result.get("error_type")
        named = f"{error_type}: " if error_type else ""
        return ReplyVerdict(
            False, True, f"server returned an error: {named}{result['error']}"
        )
    if result.get("success") is False:
        detail = result.get("error_type") or result.get("message") or result
        return ReplyVerdict(False, True, f"server reported failure: {detail}")
    return ReplyVerdict(True, True, None)


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
            files=writable_plan_files(candidate),
            detail=f"{type(exc).__name__}: {exc}",
        )

    verdict = classify_reply(result)
    if not verdict.ok:
        # BOTH failure classes land here: an explicit error/success:False
        # payload, AND an empty/unanswered reply. A write this script cannot
        # confirm succeeded IS a failed write — that is the whole of the
        # empty-reply fix, and it is why repair_one has no `answered` branch
        # the way repair_project's live re-read does.
        return RepairOutcome(
            task_id=candidate.task_id,
            tag=candidate.tag,
            disposition=FAILED,
            files=writable_plan_files(candidate),
            detail=verdict.detail,
        )
    # `files` is what was WRITTEN, not what was recovered — the summary prints
    # its length, and reporting a count the payload did not carry would be the
    # same validate-then-write divergence writable_plan_files exists to close.
    return RepairOutcome(
        task_id=candidate.task_id,
        tag=candidate.tag,
        disposition=REPAIR,
        files=writable_plan_files(candidate),
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
    4. A live ``get_task`` re-read per surviving candidate, classified by
       ``classify_reply`` and then, on an ``ok`` reply, ``classify_live_task``.
       A candidate the server EXPLAINED — ``classify_reply(...).answered`` —
       is a :data:`SKIP_MISSING`; a candidate the server never answered for
       at all is a :data:`FAILED_LIVE_READ`, because the run cannot claim to
       have examined it.
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
                    files=writable_plan_files(candidate),
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
        except Exception as exc:  # noqa: BLE001 — transport-only, by construction; see below
            # This arm can ONLY be client-side/transport. The fused-memory
            # tool layer never raises across the wire — @mcp_tool_errors
            # (fused-memory/src/fused_memory/server/tool_errors.py:44-59)
            # converts every server-side exception into an ordinary reply
            # dict — so every server-side reason (TaskNotFoundError and
            # friends) arrives as a REPLY and is classified below, leaving
            # nothing but a dead socket / refused connection / timeout to
            # reach this `except`. The batch is still NOT aborted (`continue`
            # stays); what changes is only that the candidate is now counted
            # as UNJUDGED (FAILED_LIVE_READ) rather than as a benign skip —
            # SKIP_MISSING would put it in the same bucket as a task the
            # server told us is genuinely gone, and silently exit 0.
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=FAILED_LIVE_READ,
                    files=candidate.plan_files,
                    detail=f"live re-read failed: {type(exc).__name__}: {exc}",
                )
            )
            continue

        # The server NEVER raises across the wire — @mcp_tool_errors
        # (fused-memory/src/fused_memory/server/tool_errors.py:44-59) turns every
        # exception into an ordinary reply dict. A get_task for a task that no
        # longer exists therefore arrives as
        # {"error": "No tasks found for ID(s): 77", "error_type": "TaskNotFoundError"},
        # sails past the `except` arm above, and — having no "status" key —
        # lands in SKIP_MISSING with a detail that explains nothing. Read the
        # reply's own explanation FIRST so the disposition carries the server's
        # reason, as RepairOutcome.detail's docstring promises it does.
        verdict = classify_reply(live)
        if not verdict.ok:
            # A candidate the server EXPLAINED (answered=True — an explicit
            # error/success:False marker) is a SKIP: the server successfully
            # told us the task is not there / not usable. A candidate the
            # server never answered for at all (answered=False — {} or a
            # non-dict reply, the same "wedged server ACKed with nothing
            # usable" shape classify_reply's docstring describes) is a
            # FAILURE, because the run cannot claim to have examined it — the
            # one expression below is what keeps the two branches from
            # drifting apart.
            disposition = SKIP_MISSING if verdict.answered else FAILED_LIVE_READ
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=disposition,
                    files=candidate.plan_files,
                    detail=f"live re-read: {verdict.detail}",
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

        # (5) Write — AND ONLY WHEN *apply* IS TRUE.
        #
        # The dry-run guarantee must not rest on `client is None` alone. That
        # holds for today's CLI, which passes no client without --apply, but it
        # is an invariant of the CALLER, not of this function: a caller that
        # hands in a live client with apply=False (a harness exercising gate 3,
        # say) would otherwise mutate task metadata while `format_summary`
        # stamps the report `[DRY RUN] ... NO WRITE WAS ATTEMPTED`. Reporting a
        # write as a non-write is the exact silent-mislabelling this module's
        # docstring claims to prevent, so `apply` is checked HERE, at the write
        # site, where the docstring's step 5 says it is checked.
        if not apply:
            outcomes.append(
                RepairOutcome(
                    task_id=candidate.task_id,
                    tag=candidate.tag,
                    disposition=REPAIR,
                    files=writable_plan_files(candidate),
                    detail="dry run: gate 3 passed on a live re-read; no write issued",
                )
            )
            continue

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


def _counts(result: RepairResult) -> dict[str, int]:
    """Count EVERY disposition, seeding the zero ones explicitly."""
    counts = dict.fromkeys(ALL_DISPOSITIONS, 0)
    for outcome in result.outcomes:
        counts[outcome.disposition] = counts.get(outcome.disposition, 0) + 1
    return counts


def format_summary(result: RepairResult) -> str:
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


def format_json_summary(results: list[RepairResult]) -> str:
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

# Exit codes, named rather than spelled inline so the epilog, main()'s
# docstring and the returns can never drift into disagreeing about what a
# number means. Each one denotes EXACTLY ONE outcome — that is the whole
# reason 3 and 4 exist rather than being folded into 1.
EXIT_OK = 0                    # ran; nothing failed to write
EXIT_WRITE_FAILED = 1          # at least one candidate FAILED to write
EXIT_NO_ROOT = 2               # no project root resolved to a readable tasks.db
EXIT_NOTHING_SCANNED = 3       # roots resolved but EVERY one was unreadable
EXIT_SERVER_UNREACHABLE = 4    # --apply, but the MCP server never handshook
EXIT_LIVE_READ_FAILED = 5      # the live re-read died mid-batch; candidates were left UNEXAMINED


def _transport_error_types() -> tuple[type[BaseException], ...]:
    """Exception types meaning "the server was not reachable / would not talk".

    httpx is imported LAZILY, for the same reason :func:`_make_client` imports
    the client lazily: a dry run must not depend on a transport library it
    never uses.

    ``httpx.HTTPError`` is the common base of ``ConnectError`` (nothing
    listening on the port), the timeouts, and ``HTTPStatusError`` (raised by the
    handshake's ``raise_for_status``). ``OSError`` covers the socket-level
    failures that never reach an httpx wrapper, and ``RuntimeError`` is what
    ``FusedMemoryClient._post`` itself raises when the reply is not a shape it
    can parse — a server that answers but will not handshake is, for this
    script's purposes, exactly as unusable as one that refuses the socket.
    """
    import httpx

    return (httpx.HTTPError, OSError, RuntimeError)


def _make_client(server_url: str):
    """Construct the MCP client. Imported LAZILY, on the apply path only.

    ``migrate_metadata_modules_to_files`` imports httpx at module scope, so
    importing it at the top of this module would make a dry run depend on a
    transport library it never uses.

    Subclassed because the migrate script's update_task recipe is the proven
    one — only the mode differs (merge, not replace) — and the subclass exists
    ONLY to set :attr:`~FusedMemoryClient._client_name`. It overrides nothing
    else, and must not: everything but the name is inherited.
    """
    from migrate_metadata_modules_to_files import FusedMemoryClient

    class RepairFusedMemoryClient(FusedMemoryClient):
        """FusedMemoryClient with an attributable clientInfo.name.

        ONE ATTRIBUTE, NO OVERRIDDEN METHODS — the handshake is INHERITED, so a
        parent ``protocolVersion`` bump or a new capability reaches this client
        automatically and there is no copy here to go stale. The name is
        load-bearing, not cosmetic: see :data:`CLIENT_NAME`.

        Guarded in tests/scripts/test_repair_wiped_metadata_files.py.
        """

        _client_name = CLIENT_NAME

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
            "tasks.db; 3 = roots resolved but EVERY one was unreadable, so "
            "nothing was examined; 4 = --apply, but the fused-memory MCP "
            "server could not be reached (nothing was written); 5 = the "
            "fused-memory server stopped answering mid-batch, so one or more "
            "candidates were never examined (some writes may already have "
            "landed). Never read 2, 3, 4 or 5 as a clean run -- and note that "
            "only 1 ever means a write was attempted and failed."
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
    """Repair every resolved root, and never let one bad root eat the run.

    ONE UNREADABLE ROOT IS NOT A FAILED RUN. Mirroring the audit's main()
    (audit_wiped_metadata_files.py:899-933), a ``sqlite3.Error`` from root N is
    warned to stderr, recorded, and skipped so the remaining roots still run.
    On ``--apply`` that resilience is not a nicety: without it, an unreadable
    root aborts before the summary prints, and the record of the writes already
    applied to roots 1..N-1 is LOST — the reporting-honesty failure this
    module's docstring claims to prevent. Every root that completed is
    summarised, and only "EVERY root was unreadable" is an error
    (:data:`EXIT_NOTHING_SCANNED`), so :data:`EXIT_WRITE_FAILED` keeps its one
    documented meaning.
    """
    roots = discover_project_roots(project_roots=args.project_roots)
    if not roots:
        print(
            "no project root resolvable with a readable tasks.db (checked "
            "--project-root / DASHBOARD_KNOWN_PROJECT_ROOTS / the "
            "dark-factory default); NOTHING was examined",
            file=sys.stderr,
        )
        return EXIT_NO_ROOT

    now_iso = datetime.now(UTC).isoformat()
    results: list[RepairResult] = []
    unreadable: list[str] = []

    async def _repair_root(root: str, client: _ToolClient | None) -> None:
        try:
            results.append(
                await repair_project(client, root, apply=args.apply, now_iso=now_iso)
            )
        except sqlite3.Error as exc:
            # Same wording as the audit's warning, deliberately: one phrasing
            # for one condition, so the two tools cannot describe the same
            # unreadable database differently.
            print(
                f"warning: skipping unreadable project {root}: {exc}",
                file=sys.stderr,
            )
            unreadable.append(root)

    if args.apply:
        # The client is constructed ONLY here, so a dry run genuinely never
        # dials — that inertness is what makes an accidental bare invocation
        # safe, and it is asserted by pointing --server-url at a closed port.
        transport_errors = _transport_error_types()
        async with contextlib.AsyncExitStack() as stack:
            try:
                client = await stack.enter_async_context(_make_client(args.server_url))
            except transport_errors as exc:
                # A down server is a NAMED outcome, not a traceback. The catch
                # is scoped to the handshake alone — a transport failure
                # mid-batch is already a per-task FAILED/SKIP_MISSING inside
                # repair_project — so this code can only ever mean "never
                # connected", i.e. nothing was written.
                print(
                    "error: could not reach the fused-memory MCP server at "
                    f"{args.server_url}: {type(exc).__name__}: {exc}; NOTHING "
                    "was written",
                    file=sys.stderr,
                )
                return EXIT_SERVER_UNREACHABLE
            for root in roots:
                await _repair_root(root, client)
    else:
        for root in roots:
            await _repair_root(root, None)

    if unreadable:
        print(
            f"warning: {len(unreadable)} project(s) skipped due to read errors "
            "(see warnings above); results below are incomplete",
            file=sys.stderr,
        )

    if args.json:
        print(format_json_summary(results))
    elif results:
        print("\n".join(format_summary(result) for result in results))

    if unreadable and not results:
        # Nothing was scanned at all — never report that as a clean sweep, and
        # never as a write failure either.
        print(
            "error: every resolved project was unreadable; NOTHING was "
            "examined or written (this is not a clean result)",
            file=sys.stderr,
        )
        return EXIT_NOTHING_SCANNED

    failed = sum(
        1
        for result in results
        for outcome in result.outcomes
        if outcome.disposition == FAILED
    )
    unread = sum(
        1
        for result in results
        for outcome in result.outcomes
        if outcome.disposition == FAILED_LIVE_READ
    )
    # Both counts are already fully summarised above (format_summary /
    # format_json_summary print every failed_live_read task id regardless of
    # which exit code wins) -- this precedence only decides what the ONE exit
    # code reports. A rejected write outranks an unread candidate: 1 is the
    # more actionable signal, and 5's own meaning -- "candidates were left
    # unexamined" -- does not describe a failed write.
    if failed:
        return EXIT_WRITE_FAILED
    if unread:
        return EXIT_LIVE_READ_FAILED
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = ran, nothing failed; 1 = at least one candidate FAILED to
    write; 2 = no project root resolved to a readable tasks.db; 3 = roots
    resolved but every one was unreadable; 4 = ``--apply`` could not reach the
    MCP server; 5 = the MCP server stopped answering mid-batch, after the
    handshake had already succeeded.

    EACH CODE DENOTES EXACTLY ONE OUTCOME. 2, 3 and 4 are distinct from 0 for
    the reason the audit distinguishes them: 0 would otherwise cover both
    "examined everything, nothing to do" and "examined nothing at all", and a
    consumer reading only the exit code would take a total no-op for a clean
    run. They are distinct from 1 for the mirror-image reason — 1 is the
    operator's signal that a WRITE was attempted and rejected, and folding an
    unreadable database or a down server into it would send them hunting for a
    failed write that never happened. 5 is distinct from 4 by the same
    reasoning, applied to the read path, and the asymmetry between the two is
    the operator's whole re-run decision procedure: 4 means the handshake
    never happened, so NOTHING was written; 5 means the handshake succeeded
    and the server then stopped answering, so writes may ALREADY HAVE LANDED
    and the remaining candidates are unknown.
    """
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(main_async(args))
    except sqlite3.Error as exc:
        # Only root DISCOVERY can still reach here; per-root read errors are
        # handled in main_async. Nothing was examined, so this is 3, not 1.
        print(
            f"error: could not read a project database: {exc}; NOTHING was "
            "examined",
            file=sys.stderr,
        )
        return EXIT_NOTHING_SCANNED


if __name__ == "__main__":
    sys.exit(main())
