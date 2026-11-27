# Capability manifest: one merge request per task (leaves A and B)

**PRD:** `plans/one-merge-request-per-task-prd.md` · **Sidecar:** `plans/one-merge-request-per-task-prd.capability-manifest.yaml`
**Code anchors** verified against main `27c93d6f75` (2026-11-15) and, for task 7277's unmerged code, branch
`task/7277`. Cited by symbol.

**Partial decompose (Leo, 2026-11-15).** Only leaves A and B are filed. Leaves C to G are filed later by a
pre-authorised human gate task, once the interface tasks 9152, 9182, 9201, 9193, 9144 and 8528 have landed and
their landed shape has been re-read (PRD §Pre-conditions). That later decompose extends this manifest and its
sidecar with blocks C to G and sets `signal_leaves` to F. Until then the sidecar names no signal leaf, which is
accurate: the PRD's goal signal is not yet owned by a filed task.

Every binding below is PASS. Each `grep:` pointer was resolved against main by `git grep` on `main:<path>`. Each
built-here pattern was confirmed absent on main with `git grep -E` over `orchestrator/src/orchestrator/` and
`shared/src/shared/`, so its check goes from failing to passing when its producer lands.

**G5 (B + H).** The integration gate is leaf F, not filed yet. A's and B's signals are rows 18 and 20 to 23 of the
PRD's boundary-test sketch.

**G7.** Both leaves were walked against `docs/legibility/design-invariants.md`. No waiver is recorded. Two points
were tightened in the leaf briefs and the PRD's leaf B row:

- `holds-owned-and-bounded` (leaf B): the file hold of a task waiting for its rejoin dispatch is owned by the
  rejoin stamp, which is task metadata and survives a restart. The leaf restores that hold at startup from the
  stamp, so a second restart does not drop it.
- `no-silent-fail-soft` (leaf B): the fall-through to the full pipeline (worktree or plan missing) is carried on
  the `task_rejoined_merge` event as a named outcome, not only in a log line.

**Producers outside this batch.** Task 7277 (`RecoveredMergeOwner`, `OutcomeTasks`, `Settled`,
`ClaimantSource.MERGE_REQUEST`; in progress, code on `task/7277`) and task 10093 (the recovered task's file hold;
pending, no code yet). Both are wired as real dependencies of leaf B. Leaf B's brief makes the architect verify
PRD D19's three points against task 10093 as landed and carry whatever it does not provide.

**Same-file ordering.** Leaf A makes the containment classifier in `merge_lane/landing_evidence.py` public. Task
9480 reworks the callers of that same classifier, and tasks 9201 and 9144 edit `landing_evidence.py` and
`merge_lane/types.py`. Each of the three now depends on leaf A, so A (small, no prerequisites) lands first.

## A — content relation, containment classifier and the content value
| Capability | Binding | Verdict |
|---|---|---|
| A patch-containment parse to build the classified form on | `grep:orchestrator/src/orchestrator/merge_lane/landing_evidence.py::patch_content_contained` wired: it calls the private `_classify_patch_containment` | PASS |
| An ancestry test that distinguishes "no" from "git could not answer" | `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.is_ancestor_checked` | PASS |
| The module that holds the request types | `grep:orchestrator/src/orchestrator/merge_lane/types.py::WaiterRecord` | PASS |
| `ContentRelation` and `classify_content` | built here; two mechanical greps, absent today | PASS |
| `Containment`, `classify_containment`, `range_patch_ids` | built here; two mechanical greps, absent today | PASS |
| `MergeContent` (defined only; leaf C puts it on the request) | built here; mechanical grep, absent today | PASS |
| An unreadable ref raises `ContentUnclassifiable` and is never read as a relation | rejection-mechanism, built and bound by A: boundary row 18's last fixture observes the raise | PASS |

## B — lazy restart rejoin
| Capability | Binding | Verdict |
|---|---|---|
| The recovered request's owner, its task seam and its settled kinds | producer:task-7277 upstream | PASS |
| The recovered task holds its files until its merge resolves | producer:task-10093 upstream | PASS |
| A merge-phase entry point a workflow can be started at | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._merge_and_finalise` | PASS |
| The existing merge-retry resume, whose fall-through B mirrors and whose stamp B takes precedence over | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._resume_merge_retry_if_pending` | PASS |
| What the merge phase assumes has run | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._setup_worktree_and_artifacts`, `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._recover_if_already_merged` | PASS |
| A train member still parks | `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._maybe_defer_as_train_member` | PASS |
| The hold test the owner consults | `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.is_actively_held` | PASS |
| The acquire that must become a union for a rejoining task | `grep:orchestrator/src/orchestrator/lock_table.py::ModuleLockTable.try_acquire` | PASS |
| The reloadable-key table | `grep:orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` | PASS |
| `REJOIN_ON` and `stamp_rejoin_then_pend` | built here; two mechanical greps, absent today | PASS |
| `merge_rejoin.max_wait_secs` and the `task_rejoined_merge` event | built here; two mechanical greps, absent today | PASS |
| An outcome outside `REJOIN_ON`, a third rejoin and a graceful-shutdown outcome are not rejoined | rejection-mechanism, built and bound by B: boundary rows 21 and 22 observe 7277's block, and no stamp | PASS |
