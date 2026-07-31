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

import sys
from pathlib import Path

from audit_wiped_metadata_files import (
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    WipeCandidate,
    _coerce_file_list,
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

REPAIR = "repair"                                # written (or would be, in a dry run)
SKIP_CONTRADICTED = "skip_contradicted"          # constraint 2, dropped by selection
SKIP_LOCK_LEVEL_FIDELITY = "skip_lock_level_fidelity"  # constraint 1, dropped by selection
SKIP_LOCK_CHARTER = "skip_lock_charter"          # the interceptor would reject the files
SKIP_NOT_TERMINAL = "skip_not_terminal"          # live task is not done/cancelled
SKIP_FILES_PRESENT = "skip_files_present"        # already has a scope; re-run safe
SKIP_MISSING = "skip_missing"                    # live re-read returned nothing usable
FAILED = "failed"                                # the write was attempted and errored

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
