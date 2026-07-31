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

from audit_wiped_metadata_files import (
    CONTRADICTED_REAL_MERGE_SHA,
    FIDELITY_FILE_LEVEL,
    WipeCandidate,
)


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
