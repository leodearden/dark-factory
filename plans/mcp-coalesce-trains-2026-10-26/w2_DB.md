# DB
Family: D-B: delegated owner (an in-process adopter, or a TaskWorkflow resumed at merge), cheaper derails, and do-less options

## OPTION B1a: an adopted waiter parks the MCP task at merge-deferred (new Table A edges) — foil — VERDICT reject
It needs five new Table A edges and the orchestrator re-minting a human's deferral, which the deferral gate correctly refuses. All three existing recovery writers turn the human hold into pending, and restart has no re-adoption path. This is the strand class 4953 rejected.

### Design
Admission: escalation/src/escalation/server.py::merge_request registers an in-process 'adopter' on the request's future. The server runs inside the orchestrator (orchestrator/src/orchestrator/harness.py::Harness._start_escalation_server creates it with the harness's merge queue and InFlightMergeRegistry and runs uvicorn as an asyncio task on the same loop). It sets a new MergeRequest owner flag so worker.py::SpeculativeMergeWorker._coalesce_exclusion_reason admits the request. Park: on 'superseded' the adopter writes the task's current status (blocked from release_workflow, deferred held_by_session or carried_by, pending, in-progress) straight to merge-deferred. That needs new Table A edges pending/blocked/deferred->merge-deferred. Slot check: unchanged, because the member now reads merge-deferred. Landing: mark_member_done credits merge-deferred->done as for a workflow member. Non-landing: the existing MemberRedriver writes pending, the abandoned sweep writes pending, and the delivered-checks revert writes pending. Each turns a human hold into a dispatchable row. A faithful restore needs merge-deferred->deferred and ->in-progress edges. A held_by_session record cannot be re-minted by the orchestrator: fused-memory deferral_gate._corroborate_request requires holder_pid to be a live `claude` process. Restart: the adopter dies with the process. queue_store.reconstruct_merge_request rebuilds journaled requests with no flag and no waiter, and an absorbed single has already left the journal. So nothing re-creates the adopter, and AbandonedMemberSweep re-drives the parked MCP member to pending after two 900 s passes. The hold is lost, and a carrier that 'must NEVER be dispatched' (e.g. task 7149's own description) becomes dispatchable. Submitter contract: unchanged (rule 2v waits for the orchestrator's done).

### Failure modes
- [high] **restart/journal recovery (absorbed, train unstarted or mid-verify)**: The adopter is lost. AbandonedMemberSweep sees owner NONE and re-drives the member to pending, so the hold is lost and a carrier can be dispatched to an agent.
- [high] **deferred hold**: Parking clears metadata.deferral atomically. No legal edge or authority restores it.
- [high] **capability not on main (delivered_checks)**: _revert_withheld_member writes pending, so the hold becomes a dispatch invitation.
- [medium] **pinned or blocked task**: An L2 'resume' arriving while the member is merge-deferred is DEBUG-skipped (Harness._cascade_unblock_member). close_only drops the pin, and the sweep then re-pends the member.

### Invariants
- Only a TaskWorkflow parks a task at merge-deferred (the 4953 invariant) — `Broken: Table A would gain three inbound edges, shared/src/shared/task_transitions.py::_UNION`
- Every recovery writer restores the pre-park status — `Nowhere today. It would need origin-aware branches in MemberRedriver, AbandonedMemberSweep, _revert_withheld_member and reconcile_landed_row`

### Files
- `shared/src/shared/task_transitions.py`: +3 park edges and +2 restore edges
- `fused-memory/src/fused_memory/middleware/deferral_gate.py`: orchestrator re-mint of deferral records
- HUB `escalation/src/escalation/server.py`: adopter in merge_request
- HUB `orchestrator/src/orchestrator/harness.py`: origin-aware redrive and revert
- `orchestrator/src/orchestrator/merge_deferred_recovery.py`: origin-aware recovery
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: admission and reconcile_landed_row

### Table A
+pending->merge-deferred, +blocked->merge-deferred, +deferred->merge-deferred, +merge-deferred->deferred, +merge-deferred->in-progress. Plus the recon actor subset, and OPERATIONS.md's statement that Table A refuses merge-deferred->deferred.

### Skills/docs
OPERATIONS.md (holding a member during a hand carry; abandoned members), docs/task-authoring.md (deferral lifecycle), skills/unblock, skills/merge-queue.

### Cost
6-9 tasks touching three hubs plus the deferral write-authority model. High risk: it changes a ratified invariant (4953 rejected exactly this as 'a new strand class'). Test surface: test_merge_deferred_recovery.py, test_harness_train_callbacks.py, Table A tests, and deferral gate tests.

### Code quality
Fails heuristic 10 (clear invariants informatively, redundantly, uniformly enforced): 'merge-deferred means a workflow-owned park' would split into two meanings that every reader must tell apart. Fails heuristic 11 (SPOT): the origin status would be a second copy of the hold, alongside metadata.deferral. Fails heuristic 3 (carefully factored orthogonal dimensions of variability): 'who submitted' leaks into status semantics that six modules read.

## OPTION B1b: an adopted waiter walks only legal edges and holds by pin (no Table A change) — VERDICT viable
It is buildable without Table A changes and covers carriers (11 of the 13 counterfactual singles). It buys at most about 1.5% of verify compute and pays for it with a permanent second ownership model across three hubs. Its restart story still ends at a human-resolved L1.

### Design
Admission: server.py::merge_request attaches an adopter to the request (a MergeRequest owner field, and a WaiterRecord source='adopter' in the registry). The coalescer counts an adopted request as awaited. Park, on 'superseded': the adopter snapshots the origin status and deferral into a durable metadata.merge_adoption {request_id, origin_status, deferral_kind, adopted_at}. It then walks only existing edges: blocked|deferred->pending (written with the orchestrator's claimant_run_id so the dispatch gate refuses it), pending->in-progress, in-progress->merge-deferred. It files a blocking L1 'adopted_merge' so records_pin_recovery makes the member EXTERNAL_HOLD in the abandoned sweep, including across restarts. The window is short, but the train can be dequeued before the walk ends; that is the existing DD6 race, now with three writes instead of one. Slot check: unchanged (merge-deferred). Landing: mark_member_done credits merged, and the adopter closes its L1. Non-landing (derail, dissolve, incomplete): the train callbacks must read merge_adoption. They restore merge-deferred->blocked (a legal edge) with the L1 kept as the hold, instead of pending, and re-enqueue a fresh solo MergeRequest for the member, journaled and holding a registry slot. The adopter then credits that solo landing blocked->done. The delivered-checks withhold must likewise go to blocked plus pin, not pending. Restart: the adopter is lost but the L1 pin and merge_adoption survive. A startup re-adopter (or the human via the L1) re-enqueues. Without one, the member waits on the L1 indefinitely, and that L1 lands on a human (L2) if nobody resolves it. Submitter contract: rule 2v holds (the orchestrator credits). The held_by_session hold becomes 'blocked plus L1 pin' after a non-landing, which is a different hold shape the human must close.

### Failure modes
- [medium] **restart/journal recovery (after park, before landing)**: The member sits merge-deferred, pinned by the adopter's L1. The sweep leaves it alone and nothing re-enqueues it unless a startup re-adopter is built.
- [medium] **restart/journal recovery (between CAS and the flip loop)**: reconcile_landed_row prunes the tip row (merge-deferred is a preserve status). A non-tip member's ref is a pre-rebase stacked SHA, so a re-adopter or redrive sees it not-on-main and re-merges it. Whether that resolves already_merged is unverified.
- [medium] **deferred hold**: deferred->pending clears metadata.deferral at park. After a non-landing the hold returns as blocked plus L1, not deferred.
- [medium] **cancellation**: After absorption the train is cancel-refused (train_cancel_unsupported, until task 7309). R1 notes the server's waiter-path merge_cancel bypasses the lane's coalescing refusal, so a mid-stack cancel still lands the branch.
- [low] **slug branch or task_id!=branch**: Stacking keys on worktree_base/<task_id>. A member is ejected and stays solo unless its worktree is .worktrees/<task_id>.
- [medium] **capability not on main (delivered_checks)**: Needs an adopted-aware revert (blocked plus pin). The existing revert to pending would dispatch a carrier.

### Invariants
- An adopted member's hold survives every non-landing exit as blocked plus a pinning L1, never pending — `Must be added to harness.py::build_train_callback_factory (redrive_member, _revert_withheld_member) and to worker.py's unwind. Four sites, keyed on metadata.merge_adoption.`
- An adopted request is never lost on absorption — `worker.py: keep the absorbed MergeRequest on the GroupMergeRequest and re-enqueue it on unwind. No such field today.`

### Files
- HUB `escalation/src/escalation/server.py`: adopter registration and the superseded walk
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: admission counts adopters; unwind re-enqueues adopted members; exception and stop() unwinds
- HUB `orchestrator/src/orchestrator/harness.py`: adopted-aware redrive and revert; startup re-adopter
- `orchestrator/src/orchestrator/merge_lane/types.py`: GroupMergeRequest.absorbed_requests; WaiterRecord source
- `orchestrator/src/orchestrator/merge_lane/member_redrive.py`: non-pending outcome for adopted members

### Table A
None. It uses blocked->pending, deferred->pending, pending->in-progress, in-progress->merge-deferred, merge-deferred->blocked, blocked->done.

### Skills/docs
skills/unblock and skills/unblock-low-risk (the self-stamp becomes 'wait for done'; the superseded arm is now normal for MCP), skills/merge-queue (rule 2v unchanged; new adopted-L1 note), skills/escalation-watcher (adopted_merge L1 handling), OPERATIONS.md (hand-carry hold section), docs/task-authoring.md (merge_adoption key, Tier-A blessing).

### Cost
5-7 tasks in three hubs (worker.py 22.9k, harness.py 18.8k, server.py 5.7k lines). Medium-high risk: three-write walks race the slot pre-check, and a new durable metadata key gets four readers. Test surface: test_merge_lane_coalesce_admission.py, test_merge_lane_coalesce_unwind.py, test_harness_train_callbacks.py, test_merge_deferred_recovery.py, server merge_request tests, and a restart test. Benefit: at most the lead's 1-3.5 verify host-hours per 12 days.

### Code quality
Heuristic 10: a second hold mechanism (pin plus merge_adoption) must be honoured at four separate writers, so enforcement is not uniform. Heuristic 3: 'adopted vs workflow-awaited' becomes a new dimension threaded through callbacks that today vary only by train kind. Heuristic 7 (prefer stateless interactions between modules): the server, a request-serving module, becomes a long-lived status writer. Heuristic 6 (well-defined purpose for each entity): the escalation server starts owning task lifecycle.

## OPTION B2: hand the verified branch back to a TaskWorkflow resumed at the merge phase (handback_merge) — VERDICT viable
It is clean and submitter-agnostic for /unblock-type tasks, and removes hand stamping. It cannot reach the population that would actually join trains: 11 of 13 are carriers or plan-less interactive tasks. Worth doing only if 'the workflow owns credit for /unblock merges' is wanted for its own sake, not as the train lever.

### Design
Mechanism that exists today: TaskWorkflow._drive calls _resume_merge_retry_if_pending immediately after _recover_if_already_merged. When metadata.merge_retry_pending.branch_head equals the worktree HEAD, the branch still merges cleanly onto main (GitOps.merge_tree_conflicts), and plan.json has files, it jumps straight to _merge_and_finalise. That skips plan, execute, verify and review; pre_merge_reverify is false in dark-factory-orchestrator.yaml, so no task-role verify runs either. Today only the steward's merge-phase _requeue stamps it (_stamp_merge_retry_pending). create_worktree reuses .worktrees/<id> by name without rebasing, and setup removes only a tracked .task/, so plan.json survives. New pieces: (1) a server tool handback_merge(task_id, branch_head). It checks the worktree is .worktrees/<id> on task/<id> with HEAD==branch_head, plan.files is non-empty, the status is blocked (from release_workflow) or held_by_session deferred, and no live claimant. It stamps merge_retry_pending {..., source:'handback'} and re-pends the task through the existing legal edge (blocked->pending, which is resolve_issue 'resume' / _cascade_unblock_member, or deferred->pending). (2) workflow.py: a handback-sourced stamp never falls through to the full pipeline (on any failed precondition, block with an escalation, because human work must not be re-implemented), and it is cleared at merge terminal rather than at consume, so a restart re-dispatch fast-paths again. Admission, superseded park, mark_member_done credit, re-drive, the abandoned sweep and the delivered-checks withhold all come for free: awaited_by_workflow=True. Submitter contract: no self-stamp; poll the task status until done or blocked. Cost per landing: no extra verify or review. It adds a dispatch wait for one of 24 slots, holds that slot through the queue wait (median 42.5 min solo), runs a completion memory write, and passes the workflow merge-entry gates (scope invariant, plan_files_not_touched with architect narrowing). Coverage (measured, read-only runs.db): of the 82 post-4953 MCP singles, only 14 had any prior plan/execute phase; 68 are carriers or interactive-only tasks with no plan (e.g. 7149, 7293: 'Carrier: land ...'). Of the 13 counterfactual train-eligible singles, only 2 (4503, 7174) had workflow history. Carriers and slug branches stay MCP. Covering carriers needs a plan-less 'merge-only' workflow mode that skips plan_files_not_touched, the very gate 7149 was created to escape.

### Failure modes
- [low] **restart/journal recovery (queued or mid-verify)**: Workflow semantics: the journal recovers the request without a flag; the stranded sweep re-pends the in-progress task; the re-dispatch fast-paths on the retained handback stamp and attaches as a workflow waiter. This needs the clear-at-terminal change plus B4(i) to be train-eligible again.
- [low] **pinned or blocked task**: A declared pin makes resolve_issue refuse 'resume', so the handback is refused and the submitter falls back to MCP. The open L2 is resolved at handback; any merge failure files a fresh workflow escalation.
- [low] **deferred hold**: held_by_session deferred->pending is released at handback, by intent. carried_by holds are not eligible (the carry is not in task/<id> and has no plan).
- [none] **pending/in-progress member**: A pending task with no plan is refused. In-progress with a live claimant is refused (the workflow owns it).
- [none] **slug branch or task_id!=branch**: Out of scope; stays MCP and excluded from trains as today.
- [low] **non-tip crediting**: Ordinary merge-deferred member: mark_member_done, or found_on_main via the sweep.
- [medium] **other: merge-entry gate false positive**: plan_files_not_touched or the scope invariant can block human work the MCP path would have landed (the 5846 shape). The handback then blocks with an escalation and the human falls back to MCP.
- [medium] **other: derail or dissolution after absorption**: redrive_member writes pending and the re-dispatch re-runs the pipeline (review of human work) unless B3(f) stamps merge_retry_pending on re-drive.

### Invariants
- A handed-back task is never re-implemented by an agent — `workflow.py::TaskWorkflow._resume_merge_retry_if_pending (a handback-sourced stamp blocks instead of falling through)`
- Only a TaskWorkflow parks merge-deferred (preserved) — `Unchanged: shared/src/shared/task_transitions.py::_UNION plus TaskWorkflow._handle_superseded`
- handback is refused unless HEAD==stamp, plan.files is non-empty, and no live claimant — `New server tool, validated in code rather than in skill prose`

### Files
- HUB `escalation/src/escalation/server.py`: handback_merge tool (validate, stamp, re-pend through the Table B resume or deferred->pending)
- HUB `orchestrator/src/orchestrator/workflow.py`: handback-sourced stamp: block instead of fall-through; clear at merge terminal
- `skills/unblock/SKILL.md`: blocked-task merge procedure: handback instead of merge_request plus self-stamp
- `skills/unblock-low-risk/SKILL.md`: same; the outcome becomes 'handed back', tracked by task status
- `skills/escalation-watcher/SKILL.md`: track B3 completions by task status, not request_id
- `skills/merge-queue/SKILL.md`: routing note: planned task in .worktrees/<id> means handback; carriers and slugs use MCP

### Table A
None (blocked->pending, deferred->pending, pending->in-progress, in-progress->merge-deferred all exist).

### Skills/docs
skills/unblock, skills/unblock-low-risk, skills/escalation-watcher, skills/merge-queue; OPERATIONS.md (merging section); ARCHITECTURE.md (task lifecycle: a resume-at-merge entry).

### Cost
3 tasks: server tool, workflow.py stamp semantics, skills and docs. Two hubs. Low-medium risk. Test surface: test_workflow_merge_retry_pending.py (extend), escalation server tool tests, test_release_workflow.py neighbourhood. Benefit for trains: about 2 of 13 counterfactual singles, i.e. tens of verify-minutes per 12 days. The extra dispatch latency and task-slot occupancy apply to every handback.

### Code quality
Strong on heuristic 11 (SPOT): one owner of park, credit and re-drive, with no second ownership model. Heuristic 9 (deep modules with appropriate nesting and coherent narrow interfaces): the reuse of _merge_and_finalise is a deep, already-tested seam. Risk on heuristic 3 (carefully factored orthogonal dimensions of variability) if extended to carriers: a 'merge-only, skip planning gates' mode inside the 18.9k-line workflow.py adds a dimension orthogonal to nothing. Heuristic 12 (structured data instead of meaningful strings): stamp.source must be a typed field, not a string the workflow parses.

## OPTION B3: make a coalesce-train derail cheap and safe — VERDICT recommend
(f) plus (a), (b) and (d) cut what a derail or dissolution costs every train member today: a re-review and re-dispatch per member, and E2's 18-32 h median to re-land. They are cheap, in-family, and a prerequisite for any later MCP admission. They are not by themselves a way to admit MCP singles.

### Design
Already fixed by 4953: _unwind_coalesce_train runs restore_pre_stack_tips (coalesce_unwind.py; restorable statuses merge-deferred, pending, blocked) and then MemberRedriver.redrive, on every non-done outcome including admission-halt dissolution. Left to do, as separable pieces: (a) Exceptions escaping _do_train_merge (WorktreeMissing, RuntimeError, merger error) are caught by _merger_loop's generic handlers, which only resolve the future 'blocked'. The prefix-gated one-strike and unwind sit after the call in the same try and are skipped. Wrap the coalesce branch in try/finally so it unwinds. (b) stop() sets blocked plus SHUTDOWN on buffered trains with no unwind. Dissolve unstarted coalesce trains there as the admission-halt path does. (c) Re-enqueue a non-parked member solo after a non-landing: keep each absorbed MergeRequest on the GroupMergeRequest and, for members not workflow-awaited, enqueue a fresh journaled request in the unwind. Only meaningful with an admission design. (d) One-strike only the offender: skip the mark for train_incomplete, main-red and rebase-conflict outcomes, which are not member faults. A union-verify red still marks all, since there is no attribution. (e) Credit landed non-parked members (X->done edges exist), but never revert them to pending on a withhold. Only with admission. (f) NEW and standalone: redrive_member's not-on-main arm stamps metadata.merge_retry_pending {branch_head: the member's branch tip after restore, base_sha} before writing pending. The re-dispatched workflow then fast-paths to merge through the existing _resume_merge_retry_if_pending, instead of re-running the pipeline. E2's 7484 was 'rebuilt, reviewed' after the 10-26 dissolution, and redrive_member's own comment intends 're-dispatches a fresh solo merge workflow'. Is direction 4 standalone? Not as an admission lever. With the pre-check unchanged, every train holding a non-merge-deferred MCP member deterministically fails train_incomplete; (c) and (e) only make that failure non-destructive. Pieces (a), (b), (d) and especially (f) are standalone improvements to today's workflow-only trains. About 39 post-4953 derailed-member rows plus 9 dissolved trains (E2) are re-dispatches that (f) would short-circuit. (c) and (e) are prerequisites for any future admission design (B1b or a lane-native one).

### Failure modes
- [low] **restart/journal recovery (stop mid-buffer)**: (b) unwinds unstarted trains at stop(). A mid-verify cancel still skips the unwind (BaseException); the abandoned sweep backstops workflow members.
- [low] **pending/in-progress member**: (f) stamps only when the member was merge-deferred; redrive_member's existing race guard skips a member that moved on.
- [medium] **non-tip crediting**: A non-tip member whose stacked ref is not an ancestor of main after a landed-then-unwound outcome is re-pended. With (f) its re-dispatch fast-paths to a solo merge, which should find the work already upstream. Whether that resolves already_merged rather than an empty merge is unverified.
- [low] **capability not on main (delivered_checks)**: Unchanged for workflow members (revert to pending). (e) must not reuse this revert for non-parked members.
- [low] **other: the offender re-merges solo without review**: (f) skips review of code that was already reviewed; the solo merge gate verifies every landing (task 1724), so a genuinely red member blocks solo.

### Invariants
- Every non-done exit of a coalesce train, including exceptions and stop(), hands members back — `worker.py::SpeculativeMergeWorker._merger_loop (try/finally) and ::stop`
- A re-driven member that was already reviewed is not re-reviewed — `harness.py::build_train_callback_factory redrive_member stamps; workflow.py::_resume_merge_retry_if_pending consumes (an existing seam)`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: (a) try/finally around the coalesce _do_train_merge; (b) dissolve in stop(); (d) narrower one-strike; (c) re-enqueue
- HUB `orchestrator/src/orchestrator/harness.py`: (f) stamp merge_retry_pending in redrive_member; (e) non-parked credit
- `orchestrator/src/orchestrator/merge_lane/member_redrive.py`: pass the restored tip to redrive_member for the stamp
- `orchestrator/src/orchestrator/merge_lane/types.py`: (c) GroupMergeRequest.absorbed_requests

### Table A
None.

### Skills/docs
OPERATIONS.md (abandoned members and train derail sections: a re-driven member now fast-paths to merge); ARCHITECTURE.md (merge lane, train unwind).

### Cost
Standalone (a), (b), (d), (f): 2 tasks (worker unwind robustness; redrive fast-path stamp). Low risk. Test surface: test_merge_lane_coalesce_unwind.py, test_harness_train_callbacks.py, test_merge_lane_member_redrive.py, test_workflow_merge_retry_pending.py. Admission-only (c), (e): 2 more tasks, medium risk, only worth building alongside an admission design.

### Code quality
(f) reuses one existing obligation (merge_retry_pending) rather than adding a mechanism: heuristic 11 (SPOT) and heuristic 9 (deep modules with appropriate nesting and coherent narrow interfaces). It does widen that stamp's writers from one to two, so the stamp's shape should move to a single typed helper (heuristic 12, structured data instead of meaningful strings). (a) and (b) make the 'single policy for every non-landing exit' claim in _unwind_coalesce_train's docstring true, which today it is not (heuristic 10, clear invariants informatively, redundantly, uniformly enforced).

## OPTION B4(i): do-less — admit what a workflow actually awaits (live registry fact), keep the MCP exclusion — VERDICT recommend
It is a cheap correctness improvement that keeps the 4953 invariant and fixes the flag's staleness. Its benefit is small, so it should not be sold as the answer to Leo's question.

### Design
Replace the per-request flag test in worker.py::SpeculativeMergeWorker._coalesce_exclusion_reason with 'awaited_by_workflow OR InFlightMergeRegistry.has_workflow_waiter(branch, request_id)'. Inject the registry, or a narrow workflow_awaits(req) callable built by the harness, into SpeculativeMergeWorker; today the worker receives the registry only as a parameter of the enqueue helpers. has_workflow_waiter is true while a TaskWorkflow waiter (source='workflow') is attached to the slot whose primary is this request. registry.attach mirrors the primary's terminal outcome, 'superseded' included, onto that waiter, so the workflow's _handle_superseded parks the task in-progress->merge-deferred exactly as for a flagged request. That admits two classes the flag misses: an MCP primary to which a workflow later attached, and a journal-recovered request to which a re-dispatched workflow re-attached. Do NOT journal the flag. A recovered request has no live workflow: TaskWorkflow is not re-created by recover_pending_merges, and the task sits in-progress with a dead claimant. A journaled True would therefore admit a request nobody parks and recreate train_incomplete. Re-check the predicate on each survivor just before the superseded set_result in _stack_coalesce_train. A workflow can soft-cancel and detach during the get_changed_line_ranges or stack_train_branches awaits; dropping such a survivor closes the window for flagged requests too (R1's mid-stack cancel hazard). Benefit: E1 found 5 of 13 post-4953 exclusions were recovered workflow requests, but whether a workflow had re-attached by pass time is unknown (they were excluded at 05:41Z right after a restart), so the gain is likely 0-5 singles per 12 days.

### Failure modes
- [low] **restart/journal recovery**: A recovered request stays excluded until a re-dispatched workflow attaches; then it is admitted and parked by that workflow.
- [low] **cancellation**: A detach between selection and absorption makes the request ineligible at the re-check, so it is not absorbed.
- [none] **pending/in-progress member**: Only a live workflow waiter admits; MCP-only requests remain excluded.
- [none] **slug branch or task_id!=branch**: Unaffected (no workflow waiter).

### Invariants
- A single is coalesced only if a live TaskWorkflow will park it on 'superseded' — `worker.py::SpeculativeMergeWorker._coalesce_exclusion_reason (selection) plus a re-check in ::_stack_coalesce_train (absorption)`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: workflow_awaits predicate; re-check at absorption
- HUB `orchestrator/src/orchestrator/harness.py`: inject the registry or predicate at worker construction

### Table A
None.

### Skills/docs
None required. One line in OPERATIONS.md's train section.

### Cost
1 task, about 40 lines plus tests in test_merge_lane_coalesce_admission.py. Low risk. Benefit: small, likely 0-5 singles per 12 days.

### Code quality
Heuristic 11 (SPOT): the coalescer reads the registry, the live source of 'who awaits', instead of a copied flag that goes stale on recovery. Heuristic 10 (clear invariants informatively, redundantly, uniformly enforced): the precondition is enforced at both selection and absorption. Heuristic 5 (minimum data access scopes and lifetimes): inject a narrow callable rather than the whole registry.

## OPTION B4(ii): do-less — admit only no-task members and merge-deferred carriers — VERDICT reject
Near-zero population, and it creates a new lost-request class.

### Design
Relax _coalesce_exclusion_reason to also admit (1) requests whose task_id is unknown to the scheduler (status_check already synthesises merge-deferred and mark_member_done no-ops), and (2) requests whose task is already merge-deferred (a carrier walked in-progress->merge-deferred). Both pass the slot pre-check today. Hazards: stack_train_branches addresses worktree_base/<task_id> and branch_prefix+<task_id>, so a slug branch or a worktree outside .worktrees/<id> is ejected or mis-addressed. On a non-landing, MemberRedriver no-ops for a no-task id, so the absorbed request is lost: the submitter holds a terminal 'superseded' and a train id merge_status cannot resolve, and rule 2v's 'wait for scheduler done' never resolves because there is no task. A merge-deferred carrier is re-driven to pending on a derail, making a carrier that 'must NEVER be dispatched' dispatchable. Population: post-4953 there is exactly one no-task MCP single (iact-stream-touch-units, a /warm slug), and carriers have not been walked to merge-deferred since 4953 because no longer needed. So the benefit is about zero, and it needs B3(c) to be safe.

### Failure modes
- [high] **slug branch or task_id!=branch**: Ejected at stacking, or the request is lost on derail with no recovery owner.
- [high] **restart/journal recovery**: An absorbed no-task request has left the journal and has no sweep coverage; it is lost.
- [high] **other: carrier re-driven**: Re-pended carrier becomes dispatchable to an agent.

### Invariants
- An absorbed request always has a recovery owner — `Violated for no-task ids (MemberRedriver no-op)`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: admission relaxation

### Table A
None.

### Skills/docs
skills/merge-queue rule 2v needs a no-task arm.

### Cost
1 task to build plus B3(c) to make it safe. Benefit about 0 (1 no-task single post-4953).

### Code quality
Heuristic 10: relies on the synthesis 'TRADE-OFF' in build_train_callback_factory.status_check, a documented lie ('absent means ready') that would become load-bearing. Heuristic 1 (informative names): 'merge-deferred' synthesised for things that are not tasks.

## OPTION B4(iii): do-less — keep the exclusion, spend the effort on train economics — VERDICT recommend
On the measured premise this has the larger upside: derail waste plus re-dispatch cost against about 1.5% of verify compute. It costs fewer tasks in fewer hubs and keeps 4953's invariant intact.

### Design
Keep 4953's exclusion and fix what makes today's trains lose. (1) The line-range stackability screen is dead. worker.py::SpeculativeMergeWorker._stack_coalesce_train and workflow.py's declared-train former call GitOps.get_changed_line_ranges(task_id), but its ref is diffed as 'main...<ref>' and branches are task/<id>. A bare id fails (rc 128, empty ranges, so vacuously stackable), or resolves as an unrelated short SHA (lead item 4). Pass req.branch.full_name. That restores the screen, so the selector can pick a non-overlapping partner instead of discovering conflicts at stacking or the tip rebase (3 post-4953 rebase derails). (2) B3(f): stamp merge_retry_pending on re-drive, so the 9 dissolved and 15 derailed post-4953 trains stop costing their members a full re-dispatch, re-review and multi-hour wait (E2: 7484 landed 10.8 h after dissolution). (3) Measure, then target, union-verify derails: 11 of 15 post-4953, 8 naming ratchet or baseline tests. Task 7298 (one-file-per-entry ratchet ledger plus merge driver, landed 10-21) may already have removed a share; re-measure before building attribution or a bisecting re-verify. (4) Optional: one-strike only on member-attributable outcomes (B3(d)). Admission of MCP singles stays off; the merge-queue skill's superseded arm stays relevant only for workflow-attached cases.

### Failure modes
- [low] **other: fixed screen reduces train formation**: Genuine overlaps now exclude pairs that formerly stacked cleanly (a line overlap is not always a conflict), which could lower the merge rate. Measure merged-train count before and after.
- [none] **slug branch or task_id!=branch**: Using req.branch also fixes the screen for any request whose branch is not task/<task_id>.
- [low] **restart/journal recovery**: Unchanged from today; (2) makes post-restart re-drives fast-path.

### Invariants
- The stackability screen diffs the branch actually merged — `worker.py::_stack_coalesce_train and workflow.py's declared-train former pass the full branch ref; GitOps.get_changed_line_ranges should fail loudly on rc!=0 rather than return {}`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: line-range ref fix
- HUB `orchestrator/src/orchestrator/workflow.py`: line-range ref fix in the declared-train former
- `orchestrator/src/orchestrator/git_ops.py`: get_changed_line_ranges surfaces rc!=0
- HUB `orchestrator/src/orchestrator/harness.py`: B3(f) stamp

### Table A
None.

### Skills/docs
None beyond the B3(f) OPERATIONS.md note.

### Cost
2-3 tasks: line-range fix with a loud failure; B3(f); a measurement task on union-verify derail causes after 7298. Low risk. Test surface: test_merge_queue_coalesce.py, test_workflow_train_former.py, test_harness_train_callbacks.py. Upside: post-4953 derail waste was 10.8 verify host-hours per 12 days (lead), against at most 1-3.5 h from admitting MCP singles, plus the unmeasured re-review cost of about 60 re-driven members.

### Code quality
Heuristic 10 (clear invariants informatively, redundantly, uniformly enforced): a silently-empty diff that makes a safety screen vacuous is exactly an unenforced invariant; failing loudly fixes it. Heuristic 1 (informative names): the parameter is named ref and is passed an id, so the name should be enforced by type (QueuedBranch), which also serves heuristic 12 (structured data instead of meaningful strings). Heuristic 11 (SPOT): (2) reuses the existing merge_retry_pending seam.

## REVERIFIED
- confirmed: worker.py::SpeculativeMergeWorker._coalesce_exclusion_reason returns COALESCE_EXCLUDED_NO_AWAITING_WORKFLOW on `not candidate.awaited_by_workflow` before the injectable predicate
- confirmed: worker.py::SpeculativeMergeWorker._stack_coalesce_train resolves each survivor's future MergeOutcome('superseded', superseded_by=train_id) and retires its request_id; calls get_changed_line_ranges(task_id) with the bare task id; the comment assumes every absorbed single is workflow-awaited
- confirmed: worker.py::SpeculativeMergeWorker._merger_loop runs the one-strike and _unwind_coalesce_train AFTER `outcome = await _do_train_merge(self, req)` in the same try; the WorktreeMissing / generic Exception handlers only set the future 'blocked'; the outer finally only calls _resolve_merging_requests(SHUTDOWN), so exceptions and cancels skip the unwind
- confirmed: worker.py::SpeculativeMergeWorker.stop drains lane buffers with blocked+MERGE_WORKER_SHUTDOWN_REASON and no unwind
- confirmed: worker.py::SpeculativeMergeWorker._dissolve_coalesce_train runs _unwind_coalesce_train with no one-strike mark; train_member_ids_in_custody and redrive_abandoned_members as R1 describes
- confirmed: coalesce_unwind.py::_RESTORABLE_STATUSES = {merge-deferred, pending, blocked}; member_redrive.py::MemberRedriver.redrive yields NOT_PARKED for any status != merge-deferred
- confirmed: harness.py::build_train_callback_factory status_check synthesises 'merge-deferred' for ids absent from the scheduler; redrive_member's not-on-main arm writes pending plus clear_requeue_count, with a comment saying the intent is 're-dispatches a fresh solo merge workflow'; mark_member_done goes through the delivered-checks withhold with an unconditional revert to pending
- confirmed: queue_store.py::PersistedMergeRequest has no awaited_by_workflow field and reconstruct_merge_request builds MergeRequest without it
- confirmed: types.py::InFlightMergeRegistry.has_workflow_waiter requires entry.request_id == request_id and a source=='workflow' waiter; attach() mirrors the primary's result, superseded included, onto the attached waiter's future; the worker holds no registry reference (it is a parameter of the enqueue helpers only); escalation server.py merge_cancel is the only reader of has_workflow_waiter
- confirmed: workflow.py::TaskWorkflow._submit_to_merge_queue sets awaited_by_workflow=True and attaches as WaiterRecord(source='workflow') when the branch slot is held (e.g. by an MCP primary)
- confirmed: harness.py::Harness._start_escalation_server builds create_server(..., merge_queue=self._merge_queue, harness=self, merge_inflight_registry=...) and runs uvicorn as asyncio task 'escalation-server' in-process
- confirmed: shared/src/shared/task_transitions.py::_UNION has exactly one edge into merge-deferred (in-progress) and has blocked->pending, deferred->pending, pending->in-progress, merge-deferred->blocked, and blocked/pending/deferred->done
- confirmed: merge_deferred_recovery.py::_snapshot scans only status == MERGE_DEFERRED rows; owner classes TRAIN_CUSTODY / SINGLE_REQUEST / EXTERNAL_HOLD ...
- NEW (not in the maps): workflow.py::TaskWorkflow._drive calls _resume_merge_retry_if_pending right after _recover_if_already_merged. With metadata.merge_retry_pending.branch_head == worktree HEAD, a clean GitOps.merge_tree_conflicts probe and non-empty plan.files, it jumps to _merge_and_finalise, skipping plan/execute/verify/review. It falls through to the full pipeline (clearing the stamp) otherwise. Its only writer is _stamp_merge_retry_pending from the steward _requeue merge_phase path; it is cleared at consume, and harness.py::_clear_merge_retry_pending_for_restart clears it on Table B restart
- NEW: pre_merge_reverify is false (config.py default and dark-factory-orchestrator.yaml), so a merge-phase entry runs no task-role verify; GitOps.create_worktree reuses an existing .worktrees/<id> by name without rebasing; setup removes only a git-tracked .task/
- NEW: escalation/src/escalation/action_effects.py Table B maps resume->pending (WORKFLOW_RESUME), park->blocked, restart->pending, abandon->cancelled, close_only->no change
- NEW (measured, read-only runs.db via E1's pairing logic re-run in /tmp/dB): of 82 post-4953 MCP singles (83 with one new event), 68 had no prior plan/execute phase_enter for their task and 14 had some (1 slug). Of E1's 13 counterfactual train-eligible singles, 11 had no workflow history (carriers such as 7032/7033/7149/7293/7328, interactive tasks, and the slug iact-stream-touch-units); only 4503 and 7174 had history. get_task confirms 7149 and 7293 are 'Carrier: land ...' tasks
- corrected/extended: R1's 'get_changed_line_ranges with a bare id' also applies to workflow.py's declared-train former (workflow.py line ~2120 calls get_changed_line_ranges(cid)), consistent with lead item 4

## BENEFIT
The lead's premise caps any MCP admission at about 13 singles per 12 days. Each saves 5-16 verify-minutes, for 1-3.5 host-hours in total, or at most about 1.5% of verify compute. My re-measurement sharpens who those singles are. 11 of the 13 are carriers or plan-less interactive tasks; only 2 had ever run a workflow. This decides the delegated-owner family. B2 (resume at merge through a TaskWorkflow) is the cleanest design, submitter-agnostic and with no Table A change, but it reaches at most about 2 of the 13. Reaching carriers needs a plan-less merge-only workflow mode that skips the very gate carriers exist to escape. B1b (an in-process adopter that walks only legal edges and holds the task by a pin) does reach carriers. It costs 5-7 tasks across worker.py, harness.py and server.py, plus a permanent second ownership model enforced at four writers. Its restart story still ends at a human-resolved L1. That is a poor trade for at most 1.5% of verify compute. B1a and B4(ii) are rejected. The best spend in this family keeps the 4953 exclusion and makes trains themselves cheaper: B4(iii) plus B3. That means fixing the dead line-range screen (bare-id ref), stamping merge_retry_pending on re-drive so a derailed or dissolved member fast-paths back to merge instead of being re-dispatched and re-reviewed, unwinding on exceptions and stop(), and one-striking only attributable outcomes. Add B4(i), a cheap correctness fix: admit requests a live workflow actually awaits via has_workflow_waiter, and do not journal the flag. These are about 3-4 low-risk tasks. They target post-4953 derail waste of 10.8 verify host-hours per 12 days plus the unmeasured re-review cost of about 60 re-driven members, an upside several times larger than admitting MCP singles. B3(c) and B3(e) (re-enqueue and credit non-parked members) are the reusable prerequisites if Leo still wants admission later. Honest framing for Leo: an MCP solo landing does spend a full verify slot (about 33 h per 12 days in total since 4953). But most MCP singles never wait beside a train-eligible partner, so a train could share only a sliver of that.

## OPEN QUESTIONS
- Exactly which phases does a re-dispatched, re-driven merge-deferred member re-run (execute skipped when plan steps are done? task-role verify? review?)? E2's 7484 says 'rebuilt, reviewed'; I did not trace _drive past the SIMPLE_TASK/PLAN branch. This sizes B3(f)'s saving.
- Does set_task_status(pending, claimant_run_id=<orchestrator run>) keep Scheduler._eligible_for_dispatch closed through the adopter's pending->in-progress walk (claimant_dispatch_gate_enabled, 300 s TTL)? B1b's race safety depends on it; not read.
- When a non-tip member of a train that landed before a crash is re-merged solo (a redrive not-on-main arm, or a recovered journal row), does the lane resolve it already_merged (door patch-id backstop) or produce an empty or duplicate merge? Not traced; it bears on B1b restart and B3(c).
- Are carrier worktrees at .worktrees/<carrier_id> on task/<carrier_id> at submit time? stack_train_branches requires worktree_base/<task_id>. If not, even B1b would eject carriers at stacking. merge_queued events do not record the worktree.
- How many post-4953 union-verify derails (8 of 11 name ratchet or baseline tests) predate task 7298's ratchet-ledger change (landed 10-21)? Re-measure before building train-failure attribution.
- Would fixing the line-range screen (passing the full branch ref) lower train formation or the merge rate? This needs a before/after measure; a line overlap is not always a rebase conflict.
- For B2: is _write_completion_to_memory in _merge_and_finalise appropriate for human-authored handback work, and does the workflow's plan_files_not_touched gate (the 5846 false-positive shape) make handback block often enough to negate it? Unmeasured.