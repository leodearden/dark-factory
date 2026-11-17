# One merge request per task (PRD)

Status: authored 2026-11-14, three critic passes folded. Partly decomposed 2026-11-15 on Leo's go: leaf A is
task 10215 and leaf B is task 10216; leaves C to G are filed by the pre-authorised human gate task 10217 once
the interface tasks have landed (§Decomposition plan, "Decompose record"). Rulings by Leo, 2026-11-14 (§Rulings).
Approach B+H. Leaves A and B can start when their own prerequisites land; the gate (leaf C onward) waits
on the pending interface work in §Pre-conditions, whose final shape decompose must re-verify.

**Code anchors** are cited by symbol against main `ece2f4f563` (2026-11-14) and, for task 7277's unmerged
code, branch `task/7277` tip `2673e4b11c`. Re-locate at implementation time.

Evidence: `/home/leo/.claude/spawn-briefs/merge-request-consolidation-report.md` (investigation report;
its raw extracts under `/tmp/merge-consolidation-evidence/` are volatile).

Leaves are lettered A to G. "S3 α", "S6 β1" and the like always name another PRD's task.

## Rulings

- **R1.** A task has at most one live merge request.
- **R2.** A later submit for a task with a waiting request keeps that request's place in the queue and
  updates its content to the new submit.
- **R3.** A later submit with different content is refused while a verify is in progress for the request.
- **R4.** Readings of R2 and R3, confirmed:
  - a request that has been merged speculatively but is not being verified is still "waiting": its
    content is updated and the merge is redone;
  - a submit with the same content, a clean rebase of it, or an older tip joins the request in every
    phase and is not refused;
  - a request inside a chain that is being verified counts as "verify in progress";
  - a refused workflow waits for the running verify, then resubmits if its content did not land.
- **R5.** Restart: lazy rejoin. A restored request waits under task 7277's owner; only when it fails in a
  way a workflow could fix is the task dispatched again, straight into the merge phase. This supersedes
  the "2A" recorded in the investigation report, after design scouting found tasks 7277 and 10093
  already deliver most of it.
- **R6.** Sequencing: the intake gate waits for the review-all port tasks 9152 and 9201 and is built on
  their interface.

## Goal

After this lands:

- `get_merge_queue` never shows two live requests for one task, and no task lands twice.
- A submit for a task gets one typed answer, the same on every submit path, recorded as one
  `merge_admission` event.
- A submitter told "joined" or "refused" never has its commits silently left out: when the request
  finishes it learns whether its own tip landed, and a workflow resubmits what did not.
- After a restart, a restored request that hits a conflict or a red verify is handled by the task's own
  merge-phase workflow, not by a human reading an escalation.
- A chain's links are one record per link, read by request id by both the reservation and the landing
  step.

## Background

The written design already says one live request per branch, folded at submit and at journal recovery
(`docs/prds/merge-worktree-lifecycle-integrity.md` §3 and §4 C3; `plans/async-merge-request-prd.md` §7.2
I8). The code does not deliver it:

- Both submit paths compare the new tip with `merge_lane/types.py::_InFlightEntry.snapshot_tip`. An
  ordinary first submit never records one (`merge_lane/intake.py::register_and_enqueue_merge_request`,
  `intake.py::_acquire_and_enqueue`). The comparison almost never runs.
- The workflow path then queues a second request beside the first
  (`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._submit_to_merge_queue`, the "coalesce skipped
  — tips unavailable" arm, logged at DEBUG). This was a deliberate lesser evil in commit `e9a2754cdc`:
  attaching blind could report DONE before newer commits were included.
- The `merge_request` tool attaches the caller without comparing tips
  (`intake.py::coalesce_or_enqueue_merge_request`).
- Two further submit paths skip the registry altogether (task 4163).
- At startup the journal restores requests while the startup reconcile separately resets their tasks to
  pending; the re-run workflow then submits a twin.

Measured: on 2026-11-14 one restart doubled eight tasks; task 8493 landed twice (empty merge
`1fe1c1cc23`, tree identical to its parent). Task 8402 was doubled on 2026-11-12 with no restart (a
steward's tool submit followed by the workflow's). At least ten doubled tasks since 2026-10-31.

Facts the design has to respect (all read from the code):

- **The lane merges the branch as it stands when it merges.**
  `merge_lane/worker.py::classify_and_merge` and `worker.py::build_chain` merge the branch name, not a
  recorded commit, and `TaskWorkflow._submit_to_merge_queue` moves the branch before every submit. In the
  ordinary same-branch case the branch already points at the new commits when the gate runs. An unmerged
  waiting request therefore picks them up with no help; what needs handling is a merge product built
  before the submit, and the record of which tip a product contains
  (`RealMergeItem.merged_branch_tip`, already recorded).
- A request reaches the lane through `intake.py::enqueue_merge_request` and is registered in the
  lifecycle registry only when it is drained (`merge_lane/intake_buffers.py::MergeIntake.buffer_owned_request`).
  Between the two it is invisible to a lane query.
- The existing relation test (`merge_lane/tip_relation.py::resolve_divergent`, `git cherry old new`)
  classifies a rebase onto a newer main as new content. Workflows rebase before every submit.
- A chain's reservation can expire while its tip verify continues
  (`merge_lane/chain_reservation.py::ChainReservations.expire`); the landing walk still lands the link's
  recorded merge commit.
- Today's REPLACE (`intake.py::_replace_queued_entry`) sends the new request to the back of its lane and
  cancels the old request's waiters.
- Nothing checks, when a joined request finishes, that the joiner's commits were included.
  `WaiterRecord.submitted_tip` is written and never read.
- The key is the branch. Of 6,828 queued requests in `data/orchestrator/runs.db`, 168 had a branch name
  that differs from the task id; 140 are legacy `task/<tid>` spellings and 28 are real (side branches,
  and caller-supplied pseudo task ids such as `reconciler-fix`).

## Sketch of approach

Five mechanisms. Each names its consumer (G1).

**M1. A content relation.** New `merge_lane/content_relation.py`: `ContentRelation` (SAME, REBASED, OLDER,
DIFFERENT) and `classify_content`. Consumer: the admission gate (M2) and journal recovery.

**M2. One admission gate.** New `merge_lane/admission.py::admit_merge_request`, the only function that
turns a submission into a queued request. It keys on the task and returns a typed `Admission`.
Consumers: the workflow submit, the `merge_request` tool, the stranded-verified-green reaper, and train
submission. The older entry points (`register_and_enqueue_merge_request`,
`coalesce_or_enqueue_merge_request`, which S3 α renames `dedupe_merge_request`) are deleted.

**M3. An in-place update.** A request's submit-time facts become one frozen value, `MergeContent`,
replaced by one writer. The request keeps its id, lane, queue place, first-enqueue age and waiters. A
merge product built before the update is caught before its verify launches and redone. Consumer: the
gate's UPDATED arm.

**M4. A per-submitter answer.** `admission.py::settle_submission` awaits the request and tells each
submitter whether its own tip landed. Consumers: the workflow merge phase and the `merge_request` /
`merge_status` tools.

**M5. A task claim, checked twice.** `MergeIntake` holds one index from task id to the live request that
owns it, written at admission and cleared when the request retires. The drain re-checks it. This is R1's
enforcement structure (`docs/code-quality.md` heuristic 10, clear invariants redundantly and uniformly
enforced). Consumers: the gate, journal recovery, and operators through the refusal event.

Restart (R5) adds one seam to task 7277's `RecoveredMergeOwner`: a table of outcome kinds that send the
task back to its own workflow at the merge phase instead of blocking it.

The chain clean-up follows from R1: with one request per task, the task-id and request-id derivations of
a chain's links are the same set, so they collapse into one `ChainLink` record.

## Resolved design decisions

### Identity

**D1. Key on the task, through one function.** `merge_lane/types.py::lane_task_ids(request)` returns the
task for a single-task request and every member task for a train request. It replaces
`intake_buffers.py::answers_to`, which encodes the same rule today. The gate, the claim index and journal
recovery all call it. Heuristic 11 (SPOT).

**D2. The claim is the synchronous check-and-set.** `MergeIntake.claim(request)` returns `Claimed` or
`HeldBy(state)` with no `await` inside. The index holds the request itself, keyed by each of its
`lane_task_ids`, so a holder that has not been drained yet is still reachable: `HeldBy` carries the
holder's `LiveRequestState` (state QUEUED while it is unregistered), and `update_waiting` and
`note_refused` find the request through the index.

**Live has one meaning: the request's result future is not done.** The gate's read, `claim`,
`update_waiting` (`not_live`) and the drain check (D13) all use it. Lifecycle state decides the phase of
a live request, never whether it is live; a request whose future was resolved out of band while its
token is still in the pipeline is not live. The gate claims before it enqueues, so a
request is visible to a second submit from the moment it is admitted, including before it is drained.
Two submits that both saw "no live request" across the gate's git work cannot both enqueue: the second
claim returns `HeldBy` and the gate decides again against that request.

A claim cannot leak, because its validity is read from the holder, not from a release call:

- `claim` returns `HeldBy` only while the holder is live. A holder that is not live is overwritten in the
  same synchronous step.
- `release_claim(request_id)` is called by `enqueue_claimed` on any `BaseException` before the request is
  put on the queue, and by the gate when it refuses after claiming.
- An `ItemLifecycle.on_terminal` hook (S6 C2) also releases, as tidy-up only; those hooks run later and
  swallow their own errors, so nothing depends on them.
- A request that reaches the drain unclaimed claims for itself there, silently (D13). That covers a
  submit made before the merge worker exists and every path not yet moved onto the gate.

When the coalescing pass replaces single requests with a train (`worker.py::_maybe_coalesce_waiting_singles`),
the members' claims pass to the train's request in the same step.

**D3. An update never crosses branch, request kind or task set.** A submit whose branch, kind (single or
train) or `lane_task_ids` differ from the live request's can only join (when the relation allows) or be
refused with `REFUSED_INCOMPATIBLE`. The branch stays the key of the worktree and of
`InFlightMergeRegistry`'s slot, so a submit whose branch slot is held by another task's request is also
`REFUSED_INCOMPATIBLE`, including when the submitting task has no live request of its own. Callers that
share one pseudo task id across branches are serialized by this; they should pass distinct ids.

### Content

**D4. The lane keeps merging the live branch.** No merge site changes what it merges. Pinning merges to a
recorded commit was rejected: it would change `classify_and_merge`, `build_chain`, the re-merge paths and
the sites that read the live branch on purpose (`worker.py::_already_merged_is_genuine`,
`worker.py::_unstack_before_merge`). Consequences, stated so nobody assumes otherwise:

- R3 protects the running verify. It does not pin what a later re-merge of that request contains. If a
  verifying request is re-merged (a predecessor failed, or main moved), it merges the branch as it then
  stands, including a refused submit's commits. That is the wanted result, and D10 gives every submitter
  a true answer either way.
- So that those commits are not verified under the older submit's files and module scope, a refused
  submit's `MergeContent` is kept on the request as `pending_content`. Only a REFUSED_VERIFYING submit
  leaves one (an incompatible submit's content would change the branch, against D3); the latest refusal
  wins; `note_refused` journals it and recovery applies it. It is promoted through the D11 swap at two
  existing points: when a merge takes its content snapshot (D5), and at the D12 comparison, which
  promotes first and compares second. Those two points cover a cascade re-merge, a requeue, a parked
  head and a chain link released without any lifecycle transition. It is dropped when the request
  retires.
- "What a product contains" is the tip recorded on the product at merge time, never the request's
  record.

**D5. `MergeContent` is the request's submit-time facts.** Frozen: `branch`, `tip` (the last admitted
tip), `worktree`, `task_files`, `module_configs`, `pre_rebased`, `content_seq`. It replaces
`MergeRequest.snapshot_tip` and the loose content fields. The name `content_seq` is deliberate:
`MergeRequest.generation` already exists and means something else. Each merge takes one snapshot of the
value and passes it down, so one merge never mixes old and new facts. Heuristic 8 (prefer immutable
data).

**D6. Content is compared by ancestry and patch ids.** For the live tip `old`, the submitted tip `new`
and the lane's target ref:

- SAME: equal SHAs.
- OLDER: `new` is an ancestor of `old`.
- REBASED: neither range `merge-base(target, tip)..tip` holds a merge commit, and their patch-id sets
  are equal.
- DIFFERENT: anything else. That includes added commits, dropped or reset commits, any range holding a
  merge commit when the SHAs differ, and a rebase whose conflict resolution or shifted context changed a
  patch id.

Two honest limits. A rebase over a hot file can change patch ids without changing intent; it is then
DIFFERENT and is refused during a verify. And OLDER joins by R4 without changing the request: when a
merge product already exists the lane lands that newer content; when none exists yet the lane merges the
branch as it then stands, which is the older tip if the branch itself was reset. D10 answers each
submitter truthfully in both cases. A git failure raises `ContentUnclassifiable`; the gate treats it as DIFFERENT and
emits an event (INV-11 `no-silent-fail-soft`).

`old` is the tip recorded on the request's current merge product when one exists, otherwise the last
admitted tip, `content.tip`. `LiveRequestState` exposes it as `delivering_tip`; with no product it is a
record of the last submit, not a promise of what will be merged.

**D7. Every request has a tip.** `MergeContent.tip` is a required `str`; the gate resolves the branch
tip when the caller supplies none. `_InFlightEntry.snapshot_tip` stops being read by any decision. This
closes the root gap (supersedes task 6515).

### Decision

**D8. The decision table is total and has one home.** `admission.py::decide_admission`:

| live request for the task | SAME / REBASED / OLDER | DIFFERENT, compatible (D3) | DIFFERENT, incompatible |
|---|---|---|---|
| none, tip already on the target | ALREADY_LANDED | ALREADY_LANDED | n/a |
| none, branch slot free | ENQUEUED | ENQUEUED | n/a |
| none, branch slot held by another task's request | REFUSED_INCOMPATIBLE | REFUSED_INCOMPATIBLE | n/a |
| waiting | JOINED | UPDATED | REFUSED_INCOMPATIBLE |
| frozen | JOINED | REFUSED_VERIFYING | REFUSED_INCOMPATIBLE |

A pure function over enums with an exhaustiveness test. ALREADY_LANDED is today's pre-gate fast path in
the tool door, moved inside so it emits the same event.

**D9. "Frozen" has one definition.** `merge_lane/content_phase.py::content_phase(state, *, in_live_chain,
in_frozen_train)` returns WAITING or FROZEN. A request is FROZEN when any of these holds:

- its lifecycle state is DISPATCHING, VERIFYING, GATE_REVERIFY or FINALIZING;
- it is reserved by a chain, provisionally or confirmed, which covers the time the chain is being built
  (`_deep_chain_placement` awaits `_build_deep_chain`);
- it is a link of a chain held by an in-flight entry that has not finalized, whether or not the
  reservation is still held;
- it is a member of a live train whose tip is FROZEN.

The state half is a table, `CONTENT_FROZEN_STATES`, total over the non-terminal `ItemLifecycleState`
members and test-checked. QUEUED, LANE_BUFFERED, MERGING, AWAITING_VERIFY and REDISPATCH_PARKED are
WAITING. A head parked after a gate rebase (`merge_lane/rebased_head.py::rebased_head_item`) is not
re-merged on its own; its item copies `content_seq`, so an update to it is caught by D12 and discards
that head, including one whose first verify had passed. S6's `LaneQueries` fills
`LiveRequestState.content_frozen` from this function and the writer (D11) calls the same function at
use.

**D10. The per-submitter answer.** Every waiter records the tip it submitted and the `content_seq` it
attached at. `settle_submission` returns one of:

| answer | when | workflow does |
|---|---|---|
| LANDED | the outcome is a landing or `already_merged`, and the submitter's tip is contained in the target | finish the merge phase |
| SUPERSEDED | landed, own tip not contained, and the request was updated after this waiter attached | finish the merge phase, recording both tips; no resubmit |
| NOT_INCLUDED | landed, own tip not contained, no later update | resubmit |
| FAILED_OWN | not landed, and the waiter's content is what failed: it was ENQUEUED, UPDATED or JOINED, or it was refused and its content was later promoted (the request's final `content_seq` is above the waiter's `attached_seq`) | today's failure handling, unchanged |
| FAILED_OTHER | not landed, the waiter was refused, and the request's final `content_seq` equals its `attached_seq` | resubmit its own content; never enter conflict resolution with the other content's details |
| FAILED_OTHER | the future was cancelled and the waiter is an observer (the registry mirrors a primary's cancel onto every waiter) | as above; never the operator-cancel block, which belongs to the request's own workflow |
| UNKNOWN | containment could not be determined | run `classify_containment` once more; if still unknown, block with an escalation. Never the citation-based `validate_landing_evidence`, which can accept a landing that does not contain this tip |

Containment uses the classified form in `landing_evidence.py` (leaf A makes it public): empty output on
a tip that is not an ancestor of the target is not "contained", and a git error is UNKNOWN, never
NOT_INCLUDED. Resubmits are capped at `merge_admission.max_resubmits` (default 3) per merge phase; past
the cap the workflow blocks the task with an escalation carrying every tip and request id (INV-4
`storm-escape-required`, INV-2 `structured-facts-at-failure`).

A refused workflow (R4) is attached to the verifying request as a waiter flagged `observer`. An observer
does not count toward `awaited_by_workflow`, cancel semantics or worktree disposal. A `contended` refusal
(D11) carries no waiter: the workflow backs off and submits again, counted against the same cap.

The answer must be computable after the request has retired, when its waiter records are gone. The
durable source is the event store: `merge_admission` carries the submitted tip and `attached_seq`, and
`merge_finalized` gains the request's final `content_seq`; with a containment check that is enough.

### Update

**D11. The update keeps the request and swaps one value.**
`MergeIntake.update_waiting(request_id, content, *, expected_seq)` and the `pending_content` promotion
(D4) share one private swap, the only code that replaces a live request's `MergeContent`. In one
synchronous step `update_waiting`:

1. returns `UpdateRefused(not_live | stale | frozen)` when the request is not live (D2), its `content_seq` is
   not `expected_seq`, or `content_phase` is now FROZEN;
2. swaps the content with `content_seq` raised by one; an update whose `task_files` is unknown keeps the
   request's known files;
3. re-journals the request under the same id;
4. clears the suffix-conflict tracker's signature so file-overlap ordering is recomputed
   (`suffix_graph.py::SuffixConflictTracker.signature` is lane-wide, not per request).

On `UpdateRefused`, or when `InFlightMergeRegistry.attach` finds the slot released, the gate reads the
lane again and decides again, at most three times; after that it answers `REFUSED_VERIFYING` with
`contended=True`. An AST test asserts no other assignment to a live request's content exists in
`orchestrator/src`. The door and the workflow reach the writer and the claim through public methods on
`SpeculativeMergeWorker`; nothing outside `worker.py` touches its private intake (S3 D4).

**D12. A product built before an update never reaches a verify.** Every merge product records the
`content_seq` it was built from. The comparison with the request's current `content_seq` runs
immediately after the request enters DISPATCHING and before `_launch_verify` is called, and again after
each `_remerge_at_dispatch` (`worker.py::SpeculativeMergeWorker._dispatch_item` today;
`merge_lane/verify_dispatch.py` after task 5040). From DISPATCHING on the request is FROZEN, so no update
can be accepted later; the check never has to unwind a launched verify. On a mismatch the stale commit is
recorded dead through `_record_item_commit_dead` under a new `DeadBaseSite` member, with
`notify_merger=False` as the neighbouring dispatch sites do, so requests stacked on it re-anchor through
the re-merge, and the request is merged again from its place. `build_chain` records `content_seq` per
link when it builds. An update never cancels a running verify, because D9 refuses first.

### Enforcement

**D13. The drain re-checks the claim.** `MergeIntake.buffer_owned_request` claims for a request that
arrives unclaimed, silently, and buffers it. It refuses only a request one of whose `lane_task_ids` is
`HeldBy` another live request. A re-drain of the same request (`_requeue_request`, GATE_REVERIFY back to
QUEUED) passes, and the existing future-identity test of `MergeIntake._coalesce_reentrant_drain` is
kept. A refused request is resolved through S6's
`ItemLifecycle.resolve_duplicate` with a blocked outcome of a new terminal kind `duplicate_request`
(never `already_merged`), and `merge_request_twin_refused` is emitted with both request ids, the task
and both tips. Three refusals for one task within an hour file one L1 (INV-4). The workflow treats
`duplicate_request` as "admit again", counted against the D10 cap.

The refusal ships report-only in leaf C (event emitted, request still buffered, no L1) and is armed in
leaf D, in the commit that moves the workflow onto the gate. Between the two leaves the workflow still uses the old
entry point, and an armed check would turn today's wasteful twins into blocked tasks.

**D14. Recovery uses the same key and the same relation.**
`merge_lane/queue_store.py::recover_pending_merges` groups journal records by `lane_task_ids`, keeps one
per task, and claims it. `tip_relation.py::select_recovery_winner` is re-expressed over
`lane_task_ids`: among a task's records the winner is the one with the earliest
`merge_first_enqueued_at` (it keeps the place, R2), carrying the content facts of the latest-enqueued
record on the winner's own branch; a record on another branch is dropped with an event. The others'
waiters, if any survive a restart, attach with their own tips. `decide_c3_submit_action` and
`C3SubmitAction` are deleted. The attach tables in `tip_relation.py` go with their last caller, which
task 9133 removes.

**D15. Slot hygiene is kept and runs first.** The stale-slot reaper, the disk-scan arm for foreign or
pre-restart merge worktrees and the corpse reaper, as S3 ε2 ports them, run unchanged as the gate's first
step. A slot that is held while the claim index has no live request is reaped there when stale;
otherwise its request has not been drained yet and is treated as the task's live request, joined through
the slot. A live foreign merge worktree found by the disk scan, with no slot and no claim, is answered
JOINED with no request id and `lane_unanswerable='disk_scan'`; the workflow treats that as contended.

**D16. When the lane cannot answer, join and say so.** On S3's `QueryUnavailable`, or when the lane view
is not built yet (before the worker starts), the gate joins the registry slot's request if one exists,
otherwise enqueues (claimed when the worker exists, unclaimed and self-claiming at the drain when it
does not); it sets `MergeDispatchResult.lane_unanswerable` and emits the event S3 ε2
defines. D10 makes the blind join safe.

**D17. The doors differ only in how they treat the answer.** The workflow waits and follows D10. The
`merge_request` tool returns the `Admission`: REFUSED_VERIFYING keeps today's `duplicate_in_verify`
reject code with the live request id, delivering tip and verify age; REFUSED_INCOMPATIBLE is a new reject
code naming what differs; UPDATED reports the same request id with `lane_applied: false` (an update never
changes the lane). `merge_status` takes an optional `submitted_tip` and then reports the D10 answer for
that tip. The other doors: a workflow or a train submit that gets REFUSED_INCOMPATIBLE blocks its task
with an escalation naming both branches or both task sets; the stranded-verified-green reaper, which has
no waiting workflow, leaves its marker unset on either refusal and tries again at its next sweep. The
gate carries no door flag (heuristic 3, orthogonal dimensions).

### Restart

**D18. Lazy rejoin enters the merge phase and submits afresh.** `recovered_merge_owner.py` gains
`REJOIN_ON`, the outcome kinds a workflow can fix: content conflict and red verify, expressed over S6
β1's terminal kinds (task 9144) if they are on main when leaf B is planned and over
`MergeOutcome.status` plus `block_category` otherwise, named in one place. For such an outcome the owner:

1. writes a rejoin stamp to task metadata (restored request id, outcome kind, reason, tip, time) and
   raises a separate rejoin count that the stamp's consumption does not clear;
2. then sets the task pending. The stamp is always written before the status.

`OutcomeTasks` gains the one capability this needs. A dispatched workflow that finds the stamp runs
what `TaskWorkflow._merge_and_finalise` assumes has run (`_setup_worktree_and_artifacts`, loading
`plan.json` into `self.plan`, `_recover_if_already_merged`), skips planning, implementation and review,
and enters `_merge_and_finalise`. A train member still parks through `_maybe_defer_as_train_member`. The restored request
is terminal by then, so the workflow's submit is a fresh ENQUEUED request, and its existing rebase,
conflict and red-verify handling runs on live results. No outcome is serialized into the stamp and no new
failure-handling entry point is extracted. The cost is one fresh request; for a red verify that means
one more full merge verify before the workflow's handling sees the failure. R5 accepts it.

- Fall-through: when the worktree or `plan.json` is missing, or the loaded plan lists no files, the
  stamp is cleared and the full pipeline runs, as `TaskWorkflow._resume_merge_retry_if_pending` does
  today. The `task_rejoined_merge` event carries the fall-through as a named outcome (INV-11).
- Precedence: the rejoin stamp is read before `merge_retry_pending` and clears it.
- Bound: at most two rejoins per task between landings; the third failure takes 7277's block and L1
  (INV-4). The count is cleared when the task goes done.
- A graceful-shutdown outcome is never in `REJOIN_ON` and is left alone (the defect noted on task 7277
  on 2026-11-14).

**D19. The file hold survives the rejoin; this is a contract with task 10093.** Leo's standing rule: a
lock is never released before its holder's merge lands. Leaf B requires, and decompose verifies against
10093 as landed:

- the restored hold is distinguishable from a workflow's hold, so
  `scheduler.py::Scheduler.is_actively_held` being true does not make 7277's owner return
  `WORKFLOW_HELD` and skip the outcome;
- on a rejoin the hold is kept through the failure; the dispatched workflow's acquire for the same task
  id is a union with the restored keys, not an overwrite (`lock_table.py::ModuleLockTable.try_acquire`
  overwrites today), and the restored hold retires only then;
- the hold's owner while the task waits for dispatch is the rejoin stamp. The stamp is task metadata and
  survives a restart, so the hold is restored from it at startup. When the stamp is older than
  `merge_rejoin.max_wait_secs` (default 6 h) one L1 is filed and the hold stays (INV-7
  `holds-owned-and-bounded`).

Where 10093 as landed does not provide a point, leaf B carries the change.

### Chain

**D20. One `ChainLink` record.** `ChainLink(task_id, request_id, merge_commit, content_seq)`, frozen.
`ChainResult.links` is a tuple of them. `_deep_chain_placement` reserves by `request_id`; the landing walk
(`worker.py::SpeculativeMergeWorker._land_chain_prefix` today, `merge_lane/landing.py` after task 5042)
takes each link by `request_id` and stops at a link whose `content_seq` differs from its request's.
`ChainResult.link_request_ids` and the task-id lookup through `MergeIntake.take_buffered` are deleted;
`ChainReservation.link_request_ids` stays. This lands after the gate is armed: before R1 holds, the
task-id match is a partial guard against a twin of a chain link. D9's chain clause already keeps a link
from being updated, so the walk's comparison is the second check, not the first.

### Sequencing and configuration

**D21. The gate lands after S3 ε2 (task 9193).** ε2 moves `merge_request` into
`orchestrator/mcp/merge_tools.py` and its gate onto the port. Building after it means one caller module
per door and no snapshot-dict predicates left to carry. The restart twin is closed meanwhile by 7277,
and task 10116's event measures what remains. Confirmed by Leo, 2026-11-15.

**D22. Two new configuration keys.** `merge_admission.max_resubmits` (default 3) and
`merge_rejoin.max_wait_secs` (default 21600). Both are added to `config.py::RELOADABLE_FIELDS`. Per the
standing trap, a key goes into a project's yaml only after that orchestrator has restarted on the code;
the defaults need no yaml entry.

## Contracts

```python
# merge_lane/content_relation.py (A)
class ContentRelation(Enum): SAME, REBASED, OLDER, DIFFERENT
class ContentUnclassifiable(RuntimeError):      # carries old_tip, new_tip, target, git stderr (INV-2)
async def classify_content(git_ops: GitOps, *, target: str, old_tip: str, new_tip: str) -> ContentRelation

# merge_lane/landing_evidence.py (A adds public forms of what exists privately)
async def range_patch_ids(git_ops, *, target: str, tip: str) -> RangePatchIds   # ids + has_merge_commit
class Containment(Enum): CONTAINED, NOT_CONTAINED, UNKNOWN
async def classify_containment(git_ops, *, tip: str, target: str) -> Containment

# merge_lane/types.py (A defines MergeContent; C puts it on MergeRequest; E adds ChainLink)
@dataclass(frozen=True)
class MergeContent:
    branch: QueuedBranch
    tip: str                                   # required (D7)
    worktree: Path
    task_files: tuple[str, ...] | None
    module_configs: tuple[ModuleConfig, ...]
    pre_rebased: bool
    content_seq: int = 1
def lane_task_ids(request: MergeRequest) -> frozenset[str]            # C; replaces answers_to
@dataclass(frozen=True)
class ChainLink: task_id: str; request_id: str; merge_commit: str; content_seq: int

# merge_lane/content_phase.py (C)
class RequestPhase(Enum): WAITING, FROZEN
CONTENT_FROZEN_STATES: frozenset[ItemLifecycleState]
def content_phase(state: ItemLifecycleState, *, in_live_chain: bool, in_frozen_train: bool) -> RequestPhase

# merge_lane/admission.py (C)
class AdmissionKind(Enum):
    ENQUEUED, JOINED, UPDATED, REFUSED_VERIFYING, REFUSED_INCOMPATIBLE, ALREADY_LANDED
@dataclass(frozen=True)
class Admission:
    kind: AdmissionKind
    request_id: str | None           # the live request the submitter waits on; None for ALREADY_LANDED
    waiter_id: str | None
    slot_branch: str | None          # the registry slot the waiter is attached under
    relation: ContentRelation | None
    submitted_tip: str
    delivering_tip: str | None
    attached_seq: int | None
    verify_age_secs: int | None
    contended: bool
    lane_unanswerable: str | None
    result: asyncio.Future[MergeOutcome] | None
def decide_admission(live: RequestPhase | None, relation: ContentRelation | None, *,
                     compatible: bool, slot_free: bool, already_landed: bool) -> AdmissionKind
async def admit_merge_request(submission: Submission, *, lane: Callable[[], MergeLaneQueries | None],
                              worker: Callable[[], AdmissionLane | None],   # None before the worker starts
                              registry: InFlightMergeRegistry,
                              git_ops: GitOps) -> Admission
class SubmitterAnswer(Enum): LANDED, SUPERSEDED, NOT_INCLUDED, FAILED_OWN, FAILED_OTHER, UNKNOWN
async def settle_submission(admission: Admission, *, target: str, git_ops: GitOps
                            ) -> tuple[SubmitterAnswer, MergeOutcome]

class AdmissionLane(Protocol):       # the public surface SpeculativeMergeWorker gives the gate
    def claim(self, request: MergeRequest) -> Claimed | HeldBy          # sync; HeldBy carries LiveRequestState
    def release_claim(self, request_id: str) -> None                                        # sync
    def note_refused(self, request_id: str, content: MergeContent) -> None                  # sync; D4
    def update_waiting(self, request_id: str, content: MergeContent, *, expected_seq: int
                       ) -> Updated | UpdateRefused                                         # sync
    async def enqueue_claimed(self, request: MergeRequest) -> None

# merge_lane/queries.py (C adds two fields; S3 owns the file, S6 μ's LaneQueries fills them)
class LiveRequestState: ... content_frozen: bool; delivering_tip: str

# recovered_merge_owner.py (B; file created by task 7277)
REJOIN_ON: frozenset[...]
class Settled: ... REJOINED
class OutcomeTasks(Protocol): ... async def stamp_rejoin_then_pend(self, tid: str, stamp: RejoinStamp, /) -> None
```

Import direction: `content_relation` → `landing_evidence`, `git_ops`. `content_phase` → the lifecycle
state enum only. `admission` → `content_relation`, `content_phase`, `queries`, `types`. `intake_buffers`,
`queries` and `queue_store` import `types` and `content_phase`, never `admission`. Heuristic 13 (files
make internal sense in isolation).

Invariants:

- **I1.** For every task id, at most one live request has it in `lane_task_ids`. Enforced at the claim
  (admission and recovery) and at the drain; both read the one index. A claim whose holder's future is
  done is not a claim.
- **I2.** `intake.py::enqueue_merge_request` is called only from `admission.py`, the worker's
  `enqueue_claimed`, and journal recovery. AST-checked over `orchestrator/src` and `escalation/src`, armed in leaf D.
- **I3.** `MergeIntake` is the only writer of a live request's `MergeContent`, through one private swap
  used by `update_waiting` and by the `pending_content` promotion. AST-checked.
- **I4.** `_launch_verify` is never called for a merge product whose `content_seq` differs from its
  request's.
- **I5.** A submitter is reported LANDED only when its own submitted tip is `CONTAINED` in the target.
- **I6.** `update_waiting` never changes a request's id, branch, lane, place, `merge_first_enqueued_at`
  or waiters.
- **I7.** A request refused at the drain always has its future resolved, through
  `ItemLifecycle.resolve_duplicate`.
- **I8.** An update never cancels a running verify.

Error semantics: `ContentUnclassifiable` and `QueryUnavailable` never raise out of the gate; each maps to
a documented arm (D6, D16) and an event. `UpdateRefused` and `HeldBy` are returned values.

Events: `merge_admission` (kind, relation, task, request id, both tips, door) on every submit;
`merge_request_twin_refused` (D13); `merge_content_remerged` (D12: request id, both sequence numbers,
both tips); `task_rejoined_merge` (D18: task, restored request id, outcome kind). New `EventType` members
go in `shared/src/shared/run_events.py` once S22 α (task 9766) has landed and in
`orchestrator/src/orchestrator/event_store.py` before it. The new outcome kind `duplicate_request` also
needs a row wherever outcome kinds are enumerated when leaf C is planned: `merge_lane/types.py`, its
mirror in `scripts/merge_lane_throughput.py`, `shared/src/shared/merge_outcomes.py` after task 9766, and
`dashboard/src/dashboard/data/outcome_colors.py` after task 9852.

## Boundary-test sketch

Real `SpeculativeMergeWorker`, real journal, real git repository unless a row says otherwise.

| # | Side | Scenario | Asserts |
|---|---|---|---|
| 1 | gate | second submit, same tip, request QUEUED | JOINED; one live request; queue place unchanged |
| 2 | gate | second submit, branch cleanly rebased onto a moved main, request VERIFYING | JOINED, not refused; verify not restarted |
| 3 | gate | second submit, new commit, request LANE_BUFFERED behind two others | UPDATED; same request id and place; journal holds the new tip under the old id; old waiter still attached |
| 4 | gate + worker | second submit, new commit, request AWAITING_VERIFY with a request stacked on it | UPDATED; `merge_content_remerged`; the stale commit is dead and the stacked request re-anchors; the verified tree contains the new commit |
| 5 | gate | second submit, new commit, request VERIFYING | REFUSED_VERIFYING with request id, delivering tip, verify age; verify continues |
| 6 | gate + chain | second submit, new commit, request is (a) provisionally reserved while its chain is being built, (b) a chain link whose reservation has expired while the tip verifies | REFUSED_VERIFYING in both; the walk lands the recorded merge commit |
| 7 | writer | request turns DISPATCHING between the gate's read and `update_waiting`; separately, two concurrent updates | `UpdateRefused(frozen)` then REFUSED_VERIFYING; the second update gets `stale` and is decided again |
| 8 | claim | two concurrent first submits for one task; separately, a claim whose enqueue raises, and a claim whose holder is cancelled while buffered in a halted lane | one ENQUEUED, one JOINED or UPDATED against it; one live request; no drain refusal. In the two failure cases the next submit is ENQUEUED, not `HeldBy` a dead request |
| 9 | worker | update lands during a dispatch-time re-merge; separately, on a head parked after a gate rebase | the comparison after `_remerge_at_dispatch` (or on entering DISPATCHING) catches it; `_launch_verify` is not called on the stale product |
| 10 | workflow | refused workflow (row 5); the verify lands | NOT_INCLUDED; resubmit; new request ENQUEUED; task done only after its own tip is contained |
| 11 | workflow | refused workflow (row 5); (a) a predecessor fails and the request is re-merged; (b) the request ends in a conflict; (c) the request is cancelled by an operator | (a) `pending_content` applied, the product contains the refused commits under the refused submit's files, it lands, observer gets LANDED, no resubmit; (b) FAILED_OTHER, no conflict resolution on the other content's details, resubmit is ENQUEUED; (c) FAILED_OTHER, no operator-cancel block |
| 12 | workflow | NOT_INCLUDED four times running | task blocked with one escalation carrying every tip and request id |
| 13 | answer | waiter attached at tip A; a steward's submit updates the request to B, which lands and does not contain A | waiter gets SUPERSEDED; no resubmit |
| 14 | door | steward submits through `merge_request`, then the task's workflow submits the same tip (task 8402's sequence) | JOINED; one live request; one verify |
| 15 | door | submit for a live task from another branch; train submit over a live member request with different content | REFUSED_INCOMPATIBLE naming what differs; live request unchanged |
| 16 | drain | an unclaimed request for a free task; a request for a task claimed by another live request, armed | the first self-claims and is buffered with no event; the second is not buffered; future resolved `duplicate_request`; event emitted; third in an hour files one L1. A re-drain of the same request id passes |
| 17 | recovery | journal holds two records for one task under different branch spellings | one request restored and claimed; no refusal event |
| 18 | relation | fixtures: clean rebase; rebase with conflict resolution; rebase with shifted context; added commit; dropped commit; reset to an ancestor; merge commit in range; unreadable ref | REBASED; DIFFERENT; DIFFERENT; DIFFERENT; DIFFERENT; OLDER; DIFFERENT; `ContentUnclassifiable` |
| 19 | restart | 2026-11-14 sequence: stop with a queued request, start, restored request lands | task done by 7277's owner; no second request; no task re-run |
| 20 | restart | restored request ends in a content conflict | stamp written before the status; task pending; dispatched workflow enters the merge phase without planning or review; its files stay held throughout and the acquire is a union |
| 21 | restart | restored request ends in a kind outside `REJOIN_ON`; separately, a third rejoin | 7277's block and L1 |
| 22 | restart | second graceful stop while the restored request waits | no status write, no L1, no stamp; request restored again |
| 23 | restart | rejoin stamp older than `merge_rejoin.max_wait_secs` with the task undispatched | one L1; hold kept |
| 24 | chain | chain of three | reservation and landing read the same `ChainLink` tuple by request id; no task-id lookup remains |
| 25 | paths | `submit_verified_green_merge_request` and a train submit for a task with a live request | both go through the gate; no second request |

## Pre-conditions for activating

Verified on main `ece2f4f563` unless marked. Decompose re-verifies each row (G3), wires an edge for every
pending one, and re-reads the landed shape of the interface rows before filing leaf C.

| Needed | State | Used by |
|---|---|---|
| `merge_lane/queries.py`, `MergeLaneQueries`, `LiveRequestState` | does not exist; task 9152 (S3 α) creates it | C |
| `ItemLifecycle` operative, `on_terminal`, `resolve_duplicate`, `requeue` | task 9182 (S6 ζ), pending; it depends on task 5040, which moves `_dispatch_item` into `merge_lane/verify_dispatch.py`. S6 C2 does not say what `resolve_duplicate` does for a request with no registered entry; decompose adds a note to 9182 | C |
| `LaneQueries` native implementation | task 9201 (S6 μ), pending | C |
| `merge_request` in `orchestrator/mcp/merge_tools.py`; gate and slot hygiene on the port | task 9193 (S3 ε2), pending | C (D15, D21) |
| `TerminalOutcomeRetention` removed from `intake.py` | task 4833, pending; lands before S6 ζ | C by order |
| typed terminal kinds on `MergeOutcome` | task 9144 (S6 β1) and S6 β2, pending | C (`duplicate_request`), B (falls back to status, D18) |
| `RecoveredMergeOwner`, `ClaimantSource.MERGE_REQUEST` | task 7277, pending, code on `task/7277`; shutdown defect noted on the task | B |
| recovered task holds its files | task 10093, pending, no code yet | B (D19) |
| `worker.py` chain policy settled | task 8528; every new `worker.py` task waits on it by Leo's 2026-11-10 ruling | C, E |
| auto-chain γ2 and the speculation probe deleted (removes `_maybe_auto_chain_generation`, a fifth enqueue caller) | task 9133, pending | D (I2), E |
| landing walk in `merge_lane/landing.py` | task 5042, pending | E |
| `landing_evidence.py` private containment classifier and `validate_landing_evidence` | exist | A, C |
| `RealMergeItem.merged_branch_tip` | exists | C (D6) |
| `TaskWorkflow._merge_and_finalise`, `TaskWorkflow._resume_merge_retry_if_pending` | exist | B |
| `ChainReservations`, `ChainResult.link_request_ids` | exist | E |
| loud twin event | task 10116, pending | D deletes its emission site with the twin arm |

Same-file edges decompose must also wire: tasks 9480 and 9201 on `landing_evidence.py` and
`tip_relation.py` (leaf A, leaf C); task 10118 on `intake.py`, `queue_store.py` and
`stranded_verified_green.py`; task 9935 on `intake_buffers.py` (leaves C and E).

## Cross-PRD relationship

| Other work | Direction | Mechanism | Owner |
|---|---|---|---|
| S3 `plans/review-all-s3-escalation-server-and-queue-boundary-prd.md` | consumes | `MergeLaneQueries` port; this PRD adds `content_frozen` and `delivering_tip` to `LiveRequestState` | S3 owns the port; leaf C owns the two additive fields and their test |
| S3 ε2 (9193) | replaces | ε2's `dedupe_merge_request` call in `merge_tools.py` becomes `admit_merge_request`; slot hygiene is kept (D15) | leaf C |
| S6 `plans/review-all-s6-merge-lane-lifecycle-outcome-types-and-provenance-prd.md` | consumes | `ItemLifecycle` (C2: `on_terminal`, `resolve_duplicate`), `LaneQueries` (C5), terminal kinds (C1) | S6 owns; leaf C adds the kind `duplicate_request` and fills the two new record fields in `LaneQueries` |
| `plans/priority-at-later-gates-prd.md` (9933–9941) | constrains | the pool order ranks requests by task; R1 makes "the request of a task" unique; `update_waiting` keeps `merge_first_enqueued_at`, which the order reads | this PRD; same-file edge with 9935 set at decompose |
| `plans/merge-lane-policy-prd.md` ζ (8528) | waits on | `worker.py`. Its `ChainPolicyState.suspects` lists task ids; under R1 a task id names one request, so it needs no change | 8528 first |
| S7 (9133); tasks 5040, 5042 | waits on | `worker.py` and the modules extracted from it | they land first |
| task 7277, task 10093 | extends | `RecoveredMergeOwner` gains `REJOIN_ON`; the recovered hold gains D19's contract | leaf B |
| `docs/prds/merge-worktree-lifecycle-integrity.md` §4 C3; `plans/async-merge-request-prd.md` §7.2 I8 | corrects | both describe the retired gates | leaf F adds a dated pointer to this PRD in each; neither is rewritten |
| task 6515 | supersedes | D7 | cancelled at decompose with a pointer |
| task 4163 | supersedes | I2 and row 25 | cancelled at decompose with a pointer; its `x_finding_key`, if any, moves to leaf D |
| task 10116 | follows | the DEBUG twin arm it instruments is deleted by leaf D | 10116 lands first |
| task 1641 (done) | closes the gap it left | comments deferring to 1641 are removed | leaves C and D, each in its own files |

## Decomposition plan

Seven entries: six code leaves and one dated human gate. Sizes are expected changed lines including
tests. File paths for symbols that pending tasks move are named as they are today and as they will be;
each leaf edits them where they live when it is planned.

**A. Content relation, containment classifier and the content value.** About 600 lines.
- Files: `merge_lane/content_relation.py` (new), `merge_lane/landing_evidence.py`,
  `merge_lane/types.py` (`MergeContent` only), `orchestrator/tests/test_merge_content_relation.py`.
- Delivers D6 and the public containment forms D10 needs; boundary row 18.
- Prereqs: same-file edges with tasks 9480 and 9201 on `landing_evidence.py` and with whichever S6 task
  holds `types.py`.
- Intermediate: unlocks C.

**B. Lazy restart rejoin.** About 800 lines.
- Files: `recovered_merge_owner.py`, `workflow.py`, `harness.py`, `scheduler.py`, `lock_table.py`, the
  recovered-hold module task 10093 lands, `config.py`, the event-type file,
  `orchestrator/tests/test_recovered_merge_rejoin.py`, `OPERATIONS.md`.
- Delivers D18, D19, `merge_rejoin.max_wait_secs`, the `task_rejoined_merge` event.
- Prereqs: tasks 7277 and 10093.
- Signal: boundary rows 20 to 23 pass against a real worker, journal and lock table.

**C. The admission gate, the claim, the in-place update; the tool door on the gate.** About 1,500 lines.
- Files: `merge_lane/admission.py` (new), `merge_lane/content_phase.py` (new), `merge_lane/intake.py`,
  `merge_lane/intake_buffers.py`, `merge_lane/queries.py`, `merge_lane/types.py`,
  `merge_lane/queue_store.py`, `merge_lane/tip_relation.py`, `merge_lane/worker.py` and
  `merge_lane/verify_dispatch.py` (D12 and the `AdmissionLane` methods only), `merge_lane/rebased_head.py`,
  `merge_lane/dead_base_ledger.py`, the event-type file,
  `orchestrator/mcp/merge_tools.py`, `harness.py` (lane wiring and the recovery injection), the outcome-kind
  files named under §Contracts, `orchestrator/tests/test_merge_admission.py`,
  `orchestrator/tests/merge_lane_ratchet_baseline.json`.
- Delivers M2, M3, M5 report-only, D1 to D5, D7 to D9, D11 to D17, the tool side of D10 including the
  two new `merge_finalized` fields.
- Prereqs: A; tasks 9152, 9182, 9201, 9193, 9144, 8528; the same-file edges above.
- Signal: boundary rows 1 to 9 and 13 to 17 pass (row 16 in report-only form); `merge_request` on a live
  task returns an `admission` kind; `get_merge_queue` shows one entry for it.
- Heuristic 14 (no file too large) binds on `intake.py`: the old gates' code leaves it and
  `admission.py` does not import back from it. More than 15 files: decompose re-examines whether the tool
  door and the recovery change split off cleanly.

**D. The workflow and the remaining submit paths on the gate; the drain check armed.** About 900 lines.
- Files: `workflow.py`, `stranded_verified_green.py`, `harness.py`, `config.py`,
  `merge_lane/intake.py`, `merge_lane/intake_buffers.py`, the event-type file,
  `orchestrator/tests/test_workflow_merge_admission.py`.
- Delivers the workflow side of D10, `merge_admission.max_resubmits`, the train and
  stranded-verified-green paths, deletion of the "coalesce skipped" arm and of
  `register_and_enqueue_merge_request`, I2's AST test, D13 armed.
- Prereqs: C, B (same file `workflow.py`), tasks 10116 and 9133.
- Signal: boundary rows 10 to 12, 16 (armed) and 25 pass; I2's test passes with no exemption beyond
  recovery.

**E. One `ChainLink` record; request ids on the verify-cancelled event.** About 500 lines.
- Files: `merge_lane/types.py`, `merge_lane/worker.py`, `merge_lane/landing.py`,
  `merge_lane/chain_reservation.py`, `merge_lane/intake_buffers.py`,
  `merge_lane/verify_cancellation.py`, `merge_pool_facts.py`,
  `orchestrator/tests/test_merge_queue_deep_landing.py`.
- Delivers D20 and adds the member request ids to the `merge_verify_cancelled` event, the one verify
  event task 9908 did not cover.
- Prereqs: D; tasks 8528, 9133, 5042; a same-file edge with 9935.
- Signal: boundary row 24 passes; `ChainResult` has no `link_request_ids` attribute.

**F. Integration gate and prose corrections.** About 400 lines.
- Files: `orchestrator/tests/test_one_request_per_task_integration.py`, `OPERATIONS.md`,
  `ARCHITECTURE.md`, `skills/merge-queue/SKILL.md`, `skills/merge-queue/references/two-layer-model.md`,
  the two dated pointers named in §Cross-PRD relationship.
- Replays the 2026-11-14 sequence (row 19) and task 8402's (row 14) end to end on the final code.
- Prereqs: D, E, B.
- Signal: the replay ends with one request per task, one landing per task and one verify per landing.

**G. Human gate, dated: read the twin events after seven days.** No code.
- Fires seven days after leaf F is done and the fleet has restarted on it.
- Leo, or a session he starts, reads `merge_request_twin_refused` and `merge_admission` from the event
  store: any refusal outside tests is a defect to file; a high share of REFUSED_VERIFYING with relation
  DIFFERENT on rebases is the D6 hot-file limit showing and is reported with a count.

Order: A and B are independent and can start first. C follows the interface tasks. D, E, F, G follow in
that order.

**Decompose record (2026-11-15, partial, on Leo's go).**

- Leaf A is task 10215. Tasks 9480, 9201 and 9144 now depend on it (same files; 9480 reworks the callers of
  the classifier A makes public).
- Leaf B is task 10216, depending on tasks 7277 and 10093.
- Leaves C to G are not filed. Human gate task 10217 depends on tasks 9152, 9182, 9201, 9193, 9144 and 8528
  and carries Leo's pre-authorisation to decompose them when it fires, after re-reading the landed
  interfaces. It also cancels tasks 6515 and 4163, which are parked behind it until then.
- Notes were added to tasks 9182 (`resolve_duplicate` for an unregistered request) and 9201 (a shutdown's
  "blocked" record read as a terminal answer), as §Pre-conditions and §Out of scope require.
- Manifest: `plans/one-merge-request-per-task-prd.capability-manifest.md` and its `.yaml` sidecar, blocks A
  and B only.

## Out of scope

- Releasing or reordering file locks. Leo's rule stands: locks are held until the merge lands.
- Eager rejoin at startup (a harness-launched workflow per restored request). Ruled out by R5.
- Pinning merges to a recorded commit (D4).
- The picker, the pool order and chain depth policy.
- A shutdown's final "blocked" record being readable as a terminal answer for a request that comes back
  under the same id. That is `LaneQueries.terminal_state`'s question and belongs to S6 μ (task 9201);
  decompose adds a note to that task.
- Cross-project behaviour. The gate is per orchestrator; reify gets it when its orchestrator restarts on
  the code.
- Cancelling a running verify in favour of newer content. Ruled out by R3.

## Open questions

Tactical only.

1. **Spelling of the rejoin stamp and count metadata keys.** `docs/task-authoring.md` decides whether
   they join the blessed set or take the `x_` prefix. Decide in B.
2. **Empty patch-id sets.** A branch whose commits are all on the target has an empty set on both sides;
   D8's ALREADY_LANDED arm should catch it first. Decide the residual case in A with a fixture.
3. **How `LaneQueries` learns of chain and train membership** without importing the worker. Suggested:
   two callables given at construction. Decide in C.
4. **Whether `MergeRequest` exposes content through read-only properties or its readers move to
   `request.content.<field>`.** Either way each merge takes one snapshot (D5) and I3's AST test holds.
   Decide in C.
5. **The journal's on-disk field names** for the tip and `content_seq`, and reading records written
   before this change. Decide in C.
6. **Whether leaf C splits.** See its file count note. Decide at decompose.
