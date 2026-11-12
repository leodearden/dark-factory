# DA
Family: D-A: held member. The lane owns train membership, and joining a train writes no task status.

## OPTION A1: Held-member contract, lane credits on landing (absorbed requests stay live) — VERDICT viable
This is the best design in the family. It needs no Table A edges, writes no status at join and never touches a hold on a non-landing exit. Restart reuses the existing solo-request recovery contract, and credit reuses the 2v veto. But it costs about 6 hub-heavy tasks against a measured benefit of at most about 1-3.5 verify host-hours per 12 days, and it makes interactive submitters pay train latency. Build it only if Leo accepts that trade or the population grows.

### Design
Admission. The pass in SpeculativeMergeWorker._maybe_coalesce_waiting_singles classifies each candidate by member kind, replacing the bare awaited_by_workflow test in _coalesce_exclusion_reason. The kinds are WORKFLOW (the flag is set), HELD and TASKLESS. One async readiness snapshot is taken per pass through a new factory callback (get_statuses plus deferral.stamped_at). A non-workflow single is admitted as HELD only if all of these hold: its task is blocked, deferred with a non-planning metadata.deferral record, or merge-deferred; branch == branch_prefix+task_id; worktree == worktree_base/task_id; lane is 'normal'; the existing risk predicate passes; the knob merge_train_coalesce_held_enabled is on. A single with no task row is admitted as TASKLESS on the same addressing test. Pending and in-progress tasks are excluded. Absorption. WORKFLOW members behave as today. HELD and TASKLESS members are not resolved 'superseded'. Their request moves to a new lifecycle state ABSORBED. The future, registry slot, server waiter and journal row (annotated absorbed_by_train) all stay live, and the captured token (status, stamped_at) is held in GroupMergeRequest.held_members. Slot. Step (a) becomes a per-kind readiness check. WORKFLOW must read merge-deferred. HELD must still equal its token. TASKLESS must still be absent. A failure is train_incomplete, which strikes nobody. Landing. WORKFLOW members get mark_member_done, unchanged. HELD members get credit_held_member: re-read status, then write done (kind merged, train note) only if the status is unchanged (deferred via expected_stamped_at CAS) or is now pending. Any other status (cancelled, done, in-progress, a re-stamped deferral) gets no write. When the delivered-checks guard withholds, there is no revert: the status stays put and a blocking L1 is filed. The held future then resolves 'done' with commit, train_id and credited flag. Non-landing exits (derail, incomplete, dissolve, an exception escaping _do_train_merge): one funnel restores pre-stack tips (a held member is restorable at its captured status, deferred included; the CAS and dirty guards are kept), then re-buffers each HELD/TASKLESS request solo at the lane head under the same request_id. WORKFLOW members are re-driven as today. Restart. stop() and merger-cancel resolve held futures with blocked+SHUTDOWN, and the journal row is kept, so recovery re-enqueues the request solo and the stack-base ledger un-stacks it. Before the CAS, a LandedRow is written per held member (train_id, held_from). Recovery drops and finalizes 'done' for any landed one, and reconcile_landed_row credits it from held_from. Cancel. Absorbed ids are refused train_cancel_unsupported. The server's waiter path asks the lane before cancelling, until 7309. Submitter. Polls stay on the original request_id: queued/verifying, then done. A done that carries a train_id routes the skills to 2v (no self-stamp; status decides). Solo MCP crediting is unchanged.

### Failure modes
- [low] **restart/journal recovery (train buffered or mid-verify)**: The held future is resolved blocked+SHUTDOWN by stop() or the merger-cancel finally, and the journal row is kept. Recovery re-enqueues it solo under the same request_id, and _unstack_before_merge strips the predecessor base. The submitter sees the existing restart contract.
- [medium] **restart/journal recovery (crash between CAS advance and held-member credit)**: A per-member write-ahead LandedRow carrying held_from exists. recover_pending_merges drops the journal row of a landed held member and records merge_finalized done for its request_id. reconcile_landed_row credits it when its status still equals held_from. Without this step, a non-tip member's stacked ref is not an ancestor of main, so it would be recovered as a redundant empty-delta solo.
- [low] **cancellation**: merge_cancel on an absorbed held request_id: the server's waiter path first asks the lane (new refusal check) and returns cancelled=False with code train_cancel_unsupported, so it never cancels the future or reaps a live train member's worktree. The same guard covers _coalescing_ids during stacking. Once 7309 lands, eject-and-redrive of a held member means re-buffering it solo and then cancelling normally.
- [medium] **member status change mid-flight**: Before the slot: the readiness token mismatches, so train_incomplete, unwind, and held members re-buffered solo with no strike and no verify spent. During verify (no re-check): the train lands, and credit_held_member writes done only from the captured status or pending. It skips in-progress (a live workflow owns the task), cancelled, done and a re-stamped deferral, and files an info L1. An L2 'resume' that re-pends a blocked member mid-verify can still dispatch a workflow into the stacked worktree. Solo MCP merges have the same exposure today.
- [low] **pinned or blocked task**: No status write happens at absorption, so pins, declare_pin and open L2s are untouched. The landing credit is blocked->done, which the submitter would have written anyway. resolve_issue stays the submitter's job; the skills' 'landed and already done' exit covers it.
- [low] **deferred hold**: The status stays deferred throughout. The credit deferred->done uses the expected_stamped_at CAS, which Scheduler.set_task_status must start forwarding. The exit clears metadata.deferral with a deferral_cleared audit, which is the same effect as the submitter's own stamp. A hold released or re-stamped mid-flight fails the CAS, so there is no write and an L1 is filed.
- [low] **capability not on main (delivered_checks)**: The guard withholds, so the status is not written (no revert to pending, which would release a human hold) and a blocking L1 'landed, capability unverified' is filed. The future resolves done with credited=false. The submitter's 2v veto reports landed-but-not-credited at the 20-minute ceiling.
- [low] **non-tip crediting**: Credit comes from the train landing itself (merge sha = train merge commit), never from ancestry, so a non-tip member's stale-by-rebase ref does not matter. The sha is carried on the outcome and the merge_finalized row.
- [none] **pending/in-progress member**: Excluded at admission: pending is dispatchable and in-progress may have a live or orphan workflow. Such a single merges solo as today.
- [none] **slug branch or task_id!=branch**: Excluded at admission by the addressing predicate, because stacking, restore, re-drive and queue surgery all key on task_id. The submission merges solo.
- [low] **other: exception escaping _do_train_merge**: Today the unwind is skipped. The single exit funnel must run in the merger loop's generic handlers too, or a held future stays pending forever inside a retired train. This also fixes the existing strand for workflow members.
- [high] **other: interactive submitter latency**: The submitter waits for train formation, start and verify (formation-to-start median 178 min after 4953, selection-biased) instead of the 42.5 min solo median. /unblock-low-risk will hit its polling deadline (handed_off) more often.

### Invariants
- A train lands only if every member passes the readiness rule of its own kind at the slot. — `merge_lane/held_members.py readiness rules, consumed by worker.py::_do_train_merge step (a)`
- An absorbed held request's future resolves exactly once, at train exit: done on landing, re-buffered solo on any non-landing exit, or blocked+SHUTDOWN on stop. Its journal row lives until then. — `a single train-exit funnel in held_members.py called from the landing path, _unwind_coalesce_train, the merger-loop exception handlers and stop(); queue_store _on_terminal semantics unchanged`
- The lane writes a held member's status only once (to done), and only from the captured status (CAS for deferred) or from pending. — `harness.py::build_train_callback_factory credit_held_member`
- A held member's task_id addresses the branch and worktree the lane stacks. — `admission predicate in worker.py::_maybe_coalesce_waiting_singles`
- An absorbed request cannot be cancelled until 7309. — `worker.py::_train_cancel_refusal extended to ABSORBED; escalation/server.py::merge_cancel consults it before the waiter path`
- train_incomplete never one-strikes. — `worker.py::SpeculativeMergeWorker._merger_loop strike hook`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: Admission classification and snapshot; absorption without resolution; generalised step (a); train-exit funnel calls; exception-path unwind; stop/merger-cancel held shutdown; strike policy; ABSORBED in _LEGAL_TRANSITIONS; snapshot exposes train_id; reconcile_landed_row held rows
- `orchestrator/src/orchestrator/merge_lane/held_members.py (new)`: HeldMember value type, per-kind readiness and exit policy, re-buffer, keeping the new dimension out of worker.py
- `orchestrator/src/orchestrator/merge_lane/types.py`: GroupMergeRequest.held_members; ItemLifecycleState.ABSORBED; TrainCallbacks readiness_snapshot and credit_held_member
- HUB `orchestrator/src/orchestrator/harness.py`: build_train_callback_factory: readiness snapshot (status plus deferral stamped_at) and credit_held_member (CAS, delivered-checks withhold that files an L1, no lane release)
- HUB `orchestrator/src/orchestrator/scheduler.py`: set_task_status and mark_done forward expected_stamped_at
- `orchestrator/src/orchestrator/merge_lane/coalesce_unwind.py`: Per-member restorable decision: held members are restorable at their captured status, deferred included
- `orchestrator/src/orchestrator/merge_lane/queue_store.py`: PersistedMergeRequest.absorbed_by_train (defaulted, last); recover_pending_merges drops and finalizes landed held members via the LandedOutbox
- `orchestrator/src/orchestrator/merge_lane/landed_outbox.py`: LandedRow optional train_id and held_from fields
- HUB `escalation/src/escalation/server.py`: merge_cancel waiter path consults the lane refusal; merge_status live answer for absorbed requests carries train_id; terminal done carries train_id and credited
- `orchestrator/src/orchestrator/config.py and dark-factory-orchestrator.yaml`: merge_train_coalesce_held_enabled knob, default off

### Table A
None. Uses the existing blocked->done, deferred->done and merge-deferred->done edges. No status write at join, and none on any non-landing exit.

### Skills/docs
skills/merge-queue/SKILL.md: a 'done' outcome carrying train_id is a train landing, handled by 2v (no self-stamp; status decides), and polls stay on the original request_id. skills/unblock/SKILL.md and skills/unblock-low-risk/SKILL.md: the same routing, plus a note that absorbed requests cannot be cancelled and may run past the polling deadline (handed_off). skills/escalation-watcher/SKILL.md: a handed_off request inside a train. OPERATIONS.md: held-member trains, the knob, the latency trade, and the L1 on a withheld credit. ARCHITECTURE.md merge-lane section: member kinds. The superseded/coalesce-* arm text stays, for workflow members.

### Cost
About 6 tasks. T1: admission, readiness snapshot and knob (worker.py, harness.py, types.py). T2: ABSORBED lifecycle, snapshot and merge_status train_id, merge_cancel guard (worker.py, server.py). T3: exits, i.e. landing credit and withhold, unwind re-buffer, exception and shutdown paths, strike policy (worker.py, harness.py, coalesce_unwind.py), the largest and riskiest. T4: restart, i.e. per-member LandedRow, recovery consult, reconcile (queue_store.py, landed_outbox.py, worker.py). T5: expected_stamped_at forwarding (scheduler.py, small). T6: skills and docs. Hubs hit: worker.py (22.9k lines) in T1-T4, harness.py (18.8k), server.py (5.7k), scheduler.py (10k). Test surface: coalesce pass, unwind and re-buffer, harness factory, queue_store recovery, server merge_cancel and merge_status, plus one e2e row per exit (land, derail, incomplete, dissolve, shutdown mid-verify, crash after CAS). Risk medium-high: a new lifecycle state and a new member-kind dimension in the lane's most change-dense code.

### Code quality
Gains. 10 (clear invariants): the implicit rule that every member was parked merge-deferred by a workflow becomes explicit member kinds with stated readiness. 11 (SPOT): readiness lives in one module instead of the 'merge-deferred' literal spread across step (a), MemberRedriver, _RESTORABLE_STATUSES and the sweep. 12 (structured data): held members get train_id on the outcome instead of skills parsing TIP_ID out of 'coalesce-<tip>-<hex>'. Costs. 3 (orthogonal dimensions): member kind (workflow/held/taskless) crosses six exit kinds, so it must be a per-kind policy in held_members.py, not if-chains in worker.py. 5 (minimum data scope and lifetime) and 7 (stateless interactions): the lane now holds live requests for a train's whole lifetime. 9 (deep modules, narrow interfaces): worker.snapshot, cancel refusal and the outcome shape all widen. 14 (no file too large): worker.py grows unless the new dimension goes into held_members.py, and splitting by helper imports would be cheating, so held_members.py must own the policy end to end. Tests should go through submit, merge_status and outcome, not lane internals (tests reaching internals are an interface smell).

## OPTION A2 (foil): Lane-owned park/restore of absorbed MCP members — VERDICT reject
All four structural walls are confirmed in code. The deferral record cannot be faithfully re-minted, there is no expected-status CAS, every recovery writer writes pending, and new Table A edges would be needed. Clearing them costs about 10 cross-package tasks and still leaves restore races. 4953 rejected this for the same reason, and A1 gets the same trains without any status write at join.

### Design
Admission is as in A1. At absorption, the lane writes each absorbed MCP member's task to merge-deferred, recording its origin (status, deferral record, claimant) in a lane-side or metadata 'merge hold' record. The single's future resolves 'superseded' as today, so the existing workflow contract applies unchanged: step (a) reads merge-deferred, mark_member_done credits on landing, and the delivered-checks withhold reverts to pending. On every non-landing exit the lane restores the recorded origin status instead of re-driving to pending. That covers MemberRedriver, AbandonedMemberSweep re-drive, _revert_withheld_member, reconcile_landed_row and the unwind. It then re-submits the request solo. Restart recovery reads the merge-hold record to restore. The submitter follows the existing superseded/coalesce-* arm and the 2v veto. Walls (all re-verified in code). (1) Table A's only edge into merge-deferred is in-progress->merge-deferred, so pending/blocked/deferred->merge-deferred must be added. merge-deferred's out-edges are blocked/cancelled/done/pending, so merge-deferred->deferred (and ->in-progress) must be added. Enforcement is on (fused-memory config enforce_transitions: true). (2) Every exit from deferred deletes metadata.deferral (deferral_gate.Exit.clear_keys). Only set_task_status(status='deferred', deferral=...) can re-mint it, and a held_by_session record is corroborated against a live process whose command is 'claude' (deferral_gate _HOLDER_COMMAND), which the orchestrator process can never be. So a restore needs either a new server-only deferral kind or a weaker write authority. (3) Scheduler.set_task_status has no expected-status CAS. fused-memory exposes expected_stamped_at only for rows that are currently deferred, so a restore from merge-deferred is an unconditional overwrite of whatever a human wrote meanwhile. (4) Every existing recovery writer writes pending, and an L2 'resume' arriving while the member is parked is skipped (merge-deferred is not in _RESUME_REPEND_STATUSES). (5) Each park and restore of blocked/deferred fires targeted reconciliation (STATUS_TRIGGERS) and counts toward park-stop.

### Failure modes
- [high] **restart/journal recovery (any phase)**: The member sits at merge-deferred with no workflow. AbandonedMemberSweep re-drives it to pending after 900-1800s unless taught to read the merge-hold record. A pending task is dispatchable, so a human hold becomes a dispatch.
- [high] **deferred hold**: The park destroys metadata.deferral. The restore cannot re-mint a held_by_session record (the holder must be a live claude pid), and a carried_by record needs re-validation against the carrier.
- [high] **member status change mid-flight**: A human change while parked is overwritten by the restore (no CAS). An L2 resume while parked is silently dropped.
- [medium] **capability not on main (delivered_checks)**: _revert_withheld_member writes pending, which releases the human hold, unless it is taught to restore the origin.
- [medium] **pinned or blocked task**: blocked->merge-deferred needs a new edge, and restore to blocked charges park-stop and fires reconciliation. 'restart/park/abandon' resolutions overwrite the parked row (_action_teardown_and_set_status), so the train goes incomplete.
- [low] **cancellation**: Same as today's workflow members: refused train_cancel_unsupported. The server waiter path still needs the A1 guard.
- [none] **slug branch or task_id!=branch**: Must be excluded, as in A1.

### Invariants
- Only a workflow, or the lane on absorption, parks a task merge-deferred. — `Table A (shared/src/shared/task_transitions.py::_UNION, with new edges) and the lane's park writer`
- Every exit from a lane park restores the recorded origin. — `Would have to be enforced in MemberRedriver, AbandonedMemberSweep, _revert_withheld_member, reconcile_landed_row and the unwind: five writers, one invariant (fails SPOT)`

### Files
- `shared/src/shared/task_transitions.py`: Add pending/blocked/deferred->merge-deferred and merge-deferred->deferred (and possibly ->in-progress); recon actor subset review
- `fused-memory/src/fused_memory/middleware/deferral_gate.py`: A server-only restore deferral kind, or relaxed holder corroboration for orchestrator restores
- HUB `fused-memory/src/fused_memory/server/tools.py and middleware/task_interceptor.py`: An expected-previous-status CAS on set_task_status
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: Park writes at absorption, merge-hold record, restore on every exit
- HUB `orchestrator/src/orchestrator/harness.py`: _revert_withheld_member and redrive_member restore origin; _cascade_unblock_member handles merge-deferred
- `orchestrator/src/orchestrator/merge_deferred_recovery.py`: Sweep restores origin instead of pending
- `orchestrator/src/orchestrator/merge_lane/member_redrive.py`: Origin-aware re-drive
- HUB `orchestrator/src/orchestrator/scheduler.py`: CAS parameter forwarding
- HUB `escalation/src/escalation/server.py`: merge_cancel guard

### Table A
New edges: pending->merge-deferred, blocked->merge-deferred, deferred->merge-deferred, merge-deferred->deferred, and possibly merge-deferred->in-progress. These reverse the 4953-era invariant that only a workflow parks a task, and OPERATIONS.md's statement that Table A refuses merge-deferred->deferred.

### Skills/docs
OPERATIONS.md (Table A, 'Holding a member during a hand-carry', the abandoned-member runbook); docs/task-authoring.md (deferral kinds, if a restore kind is added); skills/unblock (release_workflow and hold steps); skills/merge-queue is largely unchanged (the coalesce arm already exists).

### Cost
About 9-11 tasks across four packages (shared, fused-memory, orchestrator, escalation), including a fused-memory protocol change (status CAS) and a deferral write-authority change. Hubs: worker.py, harness.py, server.py, scheduler.py, plus fused-memory tools.py and task_interceptor.py. Risk high: a restart-only Table A change, and five recovery writers taught a new restore rule.

### Code quality
Fails 10 (invariants enforced redundantly and uniformly): 'restore the origin' must be enforced in five independent writers. Fails 11 (SPOT): the origin record duplicates the status, so two copies of truth. Fails 6 (well-defined purpose): merge-deferred stops meaning 'parked by a workflow awaiting a train' and becomes overloaded with human holds. Fails 8 (prefer immutable data): the human-owned deferral record must be destroyed and re-minted. Fails 7 (stateless interactions): the lane and fused-memory gain a cross-process restore protocol.

## OPTION A3: Held-member contract, status-write-free (submitter credits as on solo) — VERDICT viable
Simpler status story, but it regresses the dead-submitter case (nothing credits a non-tip held member) and adds a skill-only crediting rule. A1's lane credit reuses the existing 2v veto and the mark_done shape that workflow members already use.

### Design
This is identical to A1 in admission, absorption (the request stays live in ABSORBED), the per-kind slot check, the non-landing re-buffer, shutdown/restart and the cancel guard. The only difference is crediting. On landing the lane writes no task status for a held member. It resolves the held future MergeOutcome('done', commit=<train merge commit>, train_id=..., reason='landed in coalesce train X'). The submitter credits it exactly as it credits a solo MCP landing today (set_task_status done, kind merged), so the lane never writes a held member's status at all. The delivered-checks guard is not applied to held members, which matches solo MCP landings (task 3057 decision 2 leaves a self-merge unguarded). The submitter's sha derivation needs a new arm in skills/_shared/deriving-landed-sha.md. Today a polled done is recorded from the 'Merge task/<id> into main' marker, and a non-tip member has none, so the arm is 'a done carrying train_id: record the outcome's merge sha as kind merged'. That sha is an ancestor of main, which is the server's backstop check. Restart after a CAS: the per-member LandedRow and recovery finalize the request done (as A1). reconcile_landed_row keeps its RC-3 prune for blocked/deferred, because crediting is the submitter's job. Answer to 'should the lane also credit solo MCP landings?': under A3 symmetry runs the other way. Held train members are credited like solo MCP landings, so nothing changes for solos.

### Failure modes
- [medium] **restart/journal recovery (crash between CAS and resolution)**: Recovery finalizes the held request 'done' from the per-member LandedRow. If the submitter session died, nobody credits. The stranded-reconcile sweep credits a blocked task found_on_main only when its branch is ON_MAIN or GONE_WITH_MERGE_MARKER, which a non-tip stacked ref is not.
- [low] **capability not on main (delivered_checks)**: Not checked, matching solo MCP today. A train landing credits the member's own stacked commits, not a sibling's, so the 3057 attribution-by-inference concern is weaker, but the guard's coverage is not extended.
- [medium] **member status change mid-flight**: The pre-slot check is as A1. A mid-verify change is the submitter's to judge when it stamps, as for a solo merge today.
- [medium] **non-tip crediting**: Depends on the new sha-derivation arm. Without it the skill falls to the rebased-solo-landing arm or stops as unresolved.
- [none] **deferred hold**: No lane write ever happens. The submitter's own deferred->done closes the hold, as on solo.
- [low] **cancellation**: As A1 (refused until 7309; server waiter path guarded).
- [none] **slug branch or task_id!=branch**: Excluded at admission.

### Invariants
- The lane never writes a held member's task status. — `structural: no credit callback exists for the held kind in held_members.py`
- An absorbed held request resolves exactly once, at train exit (as in A1). — `the train-exit funnel in held_members.py`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: As A1, minus the credit call
- `orchestrator/src/orchestrator/merge_lane/held_members.py (new)`: As A1, without the credit policy
- `orchestrator/src/orchestrator/merge_lane/types.py`: As A1
- HUB `orchestrator/src/orchestrator/harness.py`: Readiness snapshot only
- `orchestrator/src/orchestrator/merge_lane/queue_store.py and landed_outbox.py`: As A1, without the reconcile change
- HUB `escalation/src/escalation/server.py`: As A1; terminal done carries train_id and merge_sha
- `skills/_shared/deriving-landed-sha.md`: New 'train-landed held member' arm

### Table A
None, and no lane status writes at all.

### Skills/docs
skills/merge-queue/SKILL.md §4 and skills/_shared/deriving-landed-sha.md gain a new train-landed arm; skills/unblock and skills/unblock-low-risk reference it; OPERATIONS.md and ARCHITECTURE.md as in A1.

### Cost
About 5 tasks: A1 minus the scheduler CAS forwarding and the credit-and-withhold work, plus a slightly larger skill edit. Same hubs: worker.py, server.py, harness.py (smaller change). Risk medium-high, with the same lifecycle work as A1.

### Code quality
Better than A1 on 6 (well-defined purpose): the lane merges, the submitter credits, and responsibilities do not blur. Better on 7 (stateless interactions): no lane-to-task write for held members. Worse on 11 (SPOT): crediting rules now live in two places (lane for workflow members, skill prose for held members), and the skill's sha-derivation ladder grows another arm, which is a prose rule nothing enforces mechanically. The 3 (orthogonal dimensions) and 14 (no file too large) concerns are the same as A1.

## OPTION A4 (foil): Cheap subset, admitting only TASKLESS singles and merge-deferred carriers — VERDICT reject
It buys roughly nothing. The eligible MCP population since 4953 is blocked or deferred real tasks, and carriers are obsolete. It is not even safe without A1's re-buffer machinery.

### Design
The coalescing pass admits a non-workflow single only when today's step (a) already passes it: a task_id the scheduler does not know (status_check synthesises merge-deferred), or a hand-carry carrier the human walked in-progress->merge-deferred before submitting. The addressing predicate is required too (branch == prefix+task_id, worktree == worktree_base/task_id). No slot-check change and no status writes. Absorption resolves 'superseded' as today. Landing credits a carrier through mark_member_done and no-ops for a taskless id. The catch on non-landing exits: MemberRedriver re-pends a merge-deferred carrier to pending, which makes a carrier task dispatchable. A taskless member passes the NOT_PARKED test (synthesised merge-deferred), but redrive_member no-ops for it and nothing re-submits the request, so the merge is silently dropped. Making the subset safe therefore needs A1's re-buffer and keep-the-request-live machinery anyway, which is most of A1's cost.

### Failure modes
- [high] **other: taskless member on derail/dissolve**: redrive_member no-ops, the request was already resolved superseded and its journal row removed, so the merge is lost until a human resubmits.
- [none] **pending/in-progress member**: Excluded.
- [high] **restart/journal recovery (any phase)**: Carrier: the sweep re-drives it to pending after 900-1800s. Taskless: lost.
- [none] **slug branch or task_id!=branch**: Excluded by addressing, which in practice excludes most taskless slug submissions too.

### Invariants
- Only members that already satisfy today's step (a) are admitted. — `admission predicate`

### Files
- HUB `orchestrator/src/orchestrator/merge_lane/worker.py`: Admission predicate; to be safe, also the A1 re-buffer path

### Table A
None.

### Skills/docs
OPERATIONS.md note only.

### Cost
1 task naively (unsafe), about 3-4 to make it safe. Population is about zero: no carrier has been needed since 4953, and E1's 13 eligible MCP singles were real tasks at blocked or deferred.

### Code quality
Leaves 10 (clear invariants) worse: it admits on a status synthesised for a different purpose (the harness's tolerance for unknown ids) and inherits the silent-drop of a taskless member, the hazard heuristic 10 exists to prevent.

## REVERIFIED
- worker.py::SpeculativeMergeWorker._coalesce_exclusion_reason returns COALESCE_EXCLUDED_NO_AWAITING_WORKFLOW before the injectable predicate. Confirmed.
- worker.py::SpeculativeMergeWorker._maybe_coalesce_waiting_singles: candidates must be normal-lane singles with a live future; the debounce signature is set before exclusions; _coalescing_ids is set across stacking. Confirmed.
- worker.py::SpeculativeMergeWorker._stack_coalesce_train: line ranges are fetched with a bare task_id. Absorbed survivors are resolved MergeOutcome('superseded', superseded_by=train_id) and retired unconditionally. A survivor whose future is already done (e.g. cancelled mid-stack) is still a member. Confirmed, with an addition: queue surgery drops buffer items by task_id, not request_id (buf_req.task_id not in survivor_set), so two requests sharing a task_id would both leave the buffer. This is a further reason for A1's addressing predicate.
- worker.py::_do_train_merge step (a): each member must read exactly 'merge-deferred', otherwise MergeOutcome('blocked') with TRAIN_INCOMPLETE_REASON_PREFIX. Confirmed. The check runs only once, before the tip rebase and verify; nothing re-checks before the CAS or the flip.
- harness.py::build_train_callback_factory: status_check synthesises merge-deferred for unknown ids only when err is None; mark_member_done probes existence, then the delivered-checks withhold reverts to pending unconditionally (current_statuses None), then mark_done(kind='merged', note 'train <id>'), consume and lane release; redrive_member no-ops for taskless ids. Confirmed.
- coalesce_unwind._RESTORABLE_STATUSES = {merge-deferred, pending, blocked}. 'deferred' is excluded. Restore is a CAS on the stacked tip with a DIRTY refusal. Confirmed.
- member_redrive.MemberRedriver.redrive acts only on members reading MERGE_DEFERRED; every other member is NOT_PARKED. Its branch is derived as branch_prefix+mid. Confirmed. Consequence: a taskless member reaches the redrive_member no-op and its merge is lost.
- git_ops.GitOps.stack_train_branches and _stack_member address members by worktree_base/<id> and branch_prefix+<id>; a missing worktree ejects. Confirmed.
- Table A (shared/src/shared/task_transitions.py::_UNION, enumerated by import): the only edge into merge-deferred is from in-progress; merge-deferred goes only to blocked, cancelled, done or pending; deferred goes only to blocked, cancelled, done or pending; blocked->done, deferred->done and pending->done exist. fused-memory/config/config.yaml has enforce_transitions: true. Confirmed.
- deferral_gate: every exit from deferred is Exit with clear_keys={'deferral'}, and held_by_session corroboration requires holder command 'claude' (_HOLDER_COMMAND). Confirmed. Refinement: the fused-memory set_task_status tool DOES accept expected_stamped_at (a CAS for a write leaving a deferred row), but orchestrator Scheduler.set_task_status and mark_done do not forward it. That is the one existing CAS A1 can use for deferred members.
- Same-status writes are always legal (is_legal_transition rule 2), and a done->done write goes through the interceptor's same-status path (provenance repair). Confirmed, so a submitter stamp after a lane credit is harmless.
- queue_store.PersistedMergeRequest has no awaited flag and new fields must be appended last with defaults; MergeQueueStore.record skips GroupMergeRequest; recover_pending_merges drops a record whose branch is gone, already an ancestor of main, or has a missing worktree, and otherwise re-enqueues it under the same request_id. Confirmed.
- escalation/server.py::merge_cancel: with a pending waiter it cancels rec.future directly and retires the slot, reaps the worktree and clears retention, never consulting the lane's _train_cancel_refusal. Confirmed. Under any design that keeps absorbed MCP futures live, this would reap a train member's worktree, so the guard is mandatory.
- merge_deferred_recovery: the sweep reads only rows whose status is MERGE_DEFERRED and uses task_ids_in_flight as the SINGLE_REQUEST owner. Confirmed.
- skills/merge-queue/SKILL.md rule 2v: no self-stamp on the coalesce-* arm, and landed-but-not-credited at the 20-minute ceiling. Plain 'done' in §4 stamps set_task_status done with merged+commit. Confirmed.
- The LandedOutbox is keyed by task_id and LandedRow carries (task_id, branch_tip_sha, advanced_sha, landed_at). _journal_landed_then_advance records the row before the advance. Confirmed, so per-member rows are feasible.
- dark-factory-orchestrator.yaml: merge_train_coalesce_enabled true, merge_train_max_members 3, merge_verify_breadth 'full'. Confirmed. A full-breadth verify makes R1's 'MCP task_files=None widens the train scope' hazard moot for this repo.
- Not re-verified, taken from the maps: release_workflow parks in-progress as blocked; the stranded-blocked redispatch has no merge-in-flight owner signal; _cascade_unblock_member skips merge-deferred; whether set_task_status or mark_done overwrites the metadata blob (R3 open question).

## BENEFIT
The premise in lead_notes says admitting MCP singles, at today's train performance, saves at most about 13 singles x 5-16 min, which is 1-3.5 verify host-hours per about 12 days (at most about 1.5% of verify compute). A1's admission is stricter than E1's upper bound. It requires a held status, branch and worktree addressing, the normal lane and a clean own-branch history. So the realistic population is a subset of those 13, plausibly 0.5-2.5 h per 12 days. The fraction of MCP singles whose worktree is worktree_base/<task_id> is unmeasured.

Against that, A1 costs about 6 tasks, four of them in worker.py (22.9k lines), plus harness.py, server.py and scheduler.py. It adds a new lifecycle state and a member-kind dimension that crosses six exit kinds. Every absorbed interactive submitter also trades the 42.5-min solo median for train formation-to-start latency (median 178 min, selection-biased) and for train derail exposure: 15 of the 46 trains that started after 4953 derailed, mostly on ratchet-style union-verify failures unrelated to MCP. That will push more /unblock-low-risk runs past their polling deadline (handed_off) and leave more B3 escalations stalled at L2.

Verdict: A1 is the right shape if the exclusion is to go. It needs no Table A edge, writes no status at join and never touches a hold on a non-landing exit. Restart reuses the solo recovery contract, and credit reuses the 2v veto. But on the measured premise its benefit is below its cost and risk, so it is not worth building now.

Cheaper things the family analysis surfaced help all trains and are worth doing regardless:
1. Fix the bare-id get_changed_line_ranges ref (lead note 4).
2. Run the unwind on exceptions that escape _do_train_merge.
3. Stop one-striking train_incomplete.
4. Guard merge_cancel's waiter path against the lane refusal.
5. Journal or recover the awaited flag (5 of the 13 post-4953 no_awaiting_workflow exclusions were recovered workflow requests, which may be as large a lever as MCP admission).

Revisit A1 if MCP volume rises or train merge rates improve.

## OPEN QUESTIONS
- Latency trade: will Leo accept train latency for interactive submitters? The alternative is a per-submission opt-in (a merge_request flag such as allow_train, default false) so only latency-insensitive hand carries join. That changes the escalation server's contract.
- What fraction of MCP singles have worktree == worktree_base/<task_id> and branch == task/<task_id>? Hand carries often live under .claude/worktrees/<name>. This is unmeasured and directly sizes A1's admissible population.
- Crediting owner for held members: lane credit (A1, reuses the 2v veto and mark_done, robust to a dead submitter) or submitter credit (A3, no lane status writes, needs a new sha-derivation arm)? This needs Leo's ruling.
- Delivered-checks for a held member: is 'leave the hold in place and file a blocking L1' acceptable instead of today's revert-to-pending for workflow members?
- Is there a status version for a blocked task (for example a row updated_at the scheduler can read) to make a readiness token stronger than a bare status equality? Only deferred rows have a CAS (expected_stamped_at).
- Does Scheduler.mark_done / set_task_status clobber metadata (dry_run_proposals, memory_hints) that /unblock-low-risk restores by hand after its own stamp? This was not verified, and it matters if the lane credits.
- Recovery of a non-tip member after a landed train with no per-member LandedRow: what does the solo path do with a branch whose own delta is already patch-applied on main after un-stacking (empty delta)? Not traced. A1 avoids the question by adding per-member LandedRows, but the behaviour should be known.
- Sequencing with 7309 (train cancel): build the absorbed-request cancel refusal now, or make 7309 a dependency?
- 6566 (a train whose tip is already on main credits no member) and 6450 (a status read failure during dissolve or re-drive strands members) also apply to held members. Should they be prerequisites?
- Would the 5 recovered-workflow exclusions (lead note 5) be fixed in the same stream, since they are the same gate and cheaper to fix than MCP admission?