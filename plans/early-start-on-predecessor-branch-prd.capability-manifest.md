# Capability manifest: start a dependent early on its predecessor's branch

**PRD:** `plans/early-start-on-predecessor-branch-prd.md` · **Sidecar:** `plans/early-start-on-predecessor-branch-prd.capability-manifest.yaml`
**Code anchors** verified against main `783d0ba683` (2026-11-16). Cited by symbol.

This file is rendered from the sidecar; the two carry the same bindings. Every `grep:` pointer was resolved
against main (the path exists; the dotted name was found by an AST walk of the file at main). Every mechanical
`delivered_check` was run with `git grep -E` at main and found absent, and
`shared.delivered_check_polarity.lint_delivered_checks` reported no finding.

Legend: `producer:Tn upstream` names a task of this plan in the dependency closure. `built here` is delivered by
the task itself and carries a `delivered_check`, mechanical where the plan fixes a name, manual otherwise.

**The check patterns are names the tasks must use.** The plan says its names are proposals, so each task's
description lists the patterns below as required names; a task that picked other names would leave its
dependents waiting. Checks for harness- and workflow-level behaviour are scoped to the source package
`orchestrator/src/orchestrator/` and match only use sites (`EventType.<member>`, `.<config key>`), because the
plan does not fix whether that code stays in `harness.py` or moves to a new module.

Signal leaves: T8 and T9 (T9 carries the end-to-end proof if T8 closed without a completed pair).

---

## T1: The successor relation in the lock table and the early-start dependency answer

*Intermediate; unlocks T4 and T5a (shared scheduler.py). No prerequisites.*

- `lock-table-acquires-and-blocker-views` → `grep:orchestrator/src/orchestrator/lock_table.py::ModuleLockTable.try_acquire` `grep:orchestrator/src/orchestrator/lock_table.py::ModuleLockTable.try_acquire_additional` `grep:orchestrator/src/orchestrator/lock_table.py::ModuleLockTable.blockers` `grep:orchestrator/src/orchestrator/lock_table.py::ModuleLockTable.blocking_holders` `grep:orchestrator/src/orchestrator/lock_table.py::ModuleLockTable._count_conflicts` the two acquires, the two blocker views and the holder count the one helper replaces. **PASS**.
- `deps-verdict-and-its-callers` → `grep:orchestrator/src/orchestrator/dependency_verdict.py::deps_verdict` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._deps_satisfied` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._park_hold_refusal` `grep:orchestrator/src/orchestrator/pure_gate_resolution.py::_dependencies_satisfied` wired: the three callers that must keep today's answers bit for bit. **PASS**.
- `lock-event-and-release` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._emit_lock_event` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.release` wired: the lock event writer and the release that drops the relation. **PASS**.
- `scheduler-state-snapshot` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.get_state_snapshot` `grep:shared/src/shared/scheduler_state.py::SchedulerStateSnapshot` the snapshot that lists the dependent as a co-holder. **PASS**.
- `hot-reloadable-config-groups` → `grep:orchestrator/src/orchestrator/config.py::OrchestratorConfig` `grep:orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` a submodel group is made hot-reloadable by adding it to RELOADABLE_FIELDS. **PASS**.
- `successor-relation-in-lock-table` → built here; ModuleLockTable declares and removes the successor relation, and its one holder-count helper skips the partner on a single-holder key. **PASS**.
  - delivered_check: grep present, pattern `[Ss]uccessor`, paths `orchestrator/src/orchestrator/lock_table.py`
- `successor-of-on-lock-event` → built here; lock_acquired for the dependent carries successor_of (contract, Events). **PASS**.
  - delivered_check: grep present, pattern `successor_of`, paths `orchestrator/src/orchestrator/scheduler.py`
- `early-start-config-group` → built here; the early_start group with its seven keys, enabled off in the code until T7 (contract, Config). **PASS**.
  - delivered_check: grep present, pattern `stop_streak_pause`, paths `orchestrator/src/orchestrator/config.py`
- `early-start-group-hot-reloadable` → built here; early_start added to RELOADABLE_FIELDS (contract, Config: hot-reloadable). **PASS**.
  - delivered_check: grep present, pattern `["']early_start["']`, paths `orchestrator/src/orchestrator/config.py`
- `early-start-dependency-answer` → built here; deps_verdict's opt-in input of waiting predecessors and its not-satisfied answer naming the predecessor (contract, Dependency test). **PASS**.
  - delivered_check: grep present, pattern `waiting_predecessor|WAITING_PREDECESSOR|early_start|EARLY_START`, paths `orchestrator/src/orchestrator/dependency_verdict.py`
- `callers-without-the-input-unchanged` → built here; ordinary starts, pinned starts and the reservation clean-up get today's answers (invariant 9; the dependency half of test 32). **PASS**.
  - delivered_check: manual. A behavioural equality over recorded inputs; no text marks it.

---

## T2a: Stacked worktree, the re-stack, and own work

*Intermediate; unlocks T2b, T5b. After task 7736 (in progress, same worktree-reuse path). Supersedes task 7660.*

- `stack-base-ledger` → `grep:orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._record_stack_base` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.own_delta_upstream` wired: the train path's record writer and a reader. StackBaseLedger.record never raises and returns nothing; StackBaseLedger.base_of answers None both for no record and for a git error. **PASS**.
- `create-worktree-stacking-input` → `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.create_worktree` its train argument branches an order > 0 member from the prior member's tip and writes no record (task 7660). **PASS**.
- `own-delta-rule` → `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.own_delta_upstream` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._solo_upstream` wired: the merge queue's use of the one rule that knows stack bases. **PASS**.
- `own-work-readers` → `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._own_work_base` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._review` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._context_complete_review_subject` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._detect_committed_branch_work` `grep:orchestrator/src/orchestrator/branch_base.py::own_work_base` wired: review and the scope readers all take their range from _own_work_base. **PASS**.
- `replay-that-aborts` → `grep:orchestrator/src/orchestrator/branch_stack.py::rebase_own_delta` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._rebase_off_base` wired: replays the commits after cut onto a named commit and aborts on a conflict. **PASS**.
- `worktree-reuse-routes` → `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._reattach_cold_worktree` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._reuse_warm_lane` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._rebase_requeued_onto_main` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.acquire_warm_lane` `grep:orchestrator/src/orchestrator/rebase_recovery.py::guarded_abort` the reuse routes that rebase an existing branch onto main today, the warm-worktree acquire a stacked worktree goes through, and the abort whose failure is only logged. **PASS**.
- `re-stack-in-reuse-routes` → built here; given the stack input on an existing branch each reuse route commits work in progress, cuts the own commits with the own_delta_upstream rule, replays them onto the named commit and re-records the base; a conflict, a refused start and a failed abort are told apart; a trial that touches nothing (section 5.3). **PASS**.
  - delivered_check: grep present, pattern `[Rr]estack|RESTACK|[^p]re_stack`, paths `orchestrator/src/orchestrator/`
- `stacked-create-worktree-writes-and-reads-back-record` → built here; create_worktree takes a stack-on-commit input, writes the ledger record before returning and returns the base; the train path uses it (closes task 7660). **PASS**.
  - delivered_check: manual. The input's name is the architect's; the T2 signal (the ledger answers the base of a branch create_worktree stacked) proves it.
- `ledger-read-tells-no-record-from-unreadable` → built here; a read of the record that answers no record, a base, or cannot read, for the guarding readers (section 5.3). **PASS**.
  - delivered_check: manual. Needed because StackBaseLedger.base_of folds a git error into None today; the name is the architect's.
- `own-work-through-stack-rule` → built here; _own_work_base answers through own_delta_upstream when the branch has a ledger record (test 28). **PASS**.
  - delivered_check: manual. own_delta_upstream already appears in workflow.py, so a grep would be green before T2 lands.
- `stacked-worktree-through-warm-pool` → built here; a stacked worktree comes from the warm-worktree pool where the project has one, with the record written and read back on that route; declared-train successors stay out (section 5.3). **PASS**.
  - delivered_check: manual. Behavioural; tests 50 and 47 (iii) prove it.
- `reuse-routes-un-stack-without-stack-input` → built here; without a stack input all three reuse routes un-stack before a rebase onto main, the registered cold worktree included. **PASS**.
  - delivered_check: manual. Behavioural; proved on a real repository.

---

## T2b: The landed question and the replay that pauses

*Intermediate; unlocks T6a, T6b, T6c. After T2a (same files).*

- `containment-classifier` → `grep:orchestrator/src/orchestrator/merge_lane/landing_evidence.py::classify_containment` `grep:orchestrator/src/orchestrator/merge_lane/landing_evidence.py::patch_content_contained` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.is_ancestor_checked` classify_containment is public and tested (task 10215) but has no production caller yet; T2 is its first consumer. Its git cherry core is wired through patch_content_contained. **PASS**.
- `ledger-and-own-commits` → producer:T2a upstream: the three-valued ledger read and the cut of own commits. **PASS**.
- `landed-question` → built here; one function answering landed as built on / landed, changed / not landed / unknown, taking the predecessor's recorded landings and main's tip at the early start as inputs (section 5.6). **PASS**.
  - delivered_check: grep present, pattern `[Ll]anded_?[Cc]hanged|LANDED_CHANGED`, paths `orchestrator/src/orchestrator/`
- `replay-that-pauses` → built here; replay a branch's own commits onto main, pause at a conflict, let the caller resolve and continue, drop the record only when complete and checked (contract, Worktree and git). **PASS**.
  - delivered_check: manual. The entry point name is the architect's; the T2b signal (a paused replay is continued and only then drops the record) proves it.

---

## T3: The borrowed slot

*Intermediate; unlocks T5b. No prerequisites. Keep the slot-ended signal in slot_lease.py so T3 stays off scheduler.py.*

- `slot-lease-and-its-exits` → `grep:orchestrator/src/orchestrator/slot_lease.py::SlotLease.release` `grep:orchestrator/src/orchestrator/harness.py::Harness._run_slot` wired: _run_slot builds the lease and releases it on every exit. **PASS**.
- `harness-waits-for-a-permit` → `grep:orchestrator/src/orchestrator/harness.py::Harness.run` `grep:orchestrator/src/orchestrator/harness.py::Harness._launch_slot` wired: run takes acquire_next's task, then waits on the semaphore, then launches. **PASS**.
- `borrowing-lease` → built here; a lease that takes no permit and names its lender; the lender's release hands its permit to a still-running borrower (contract, SlotLease). **PASS**.
  - delivered_check: grep present, pattern `[Ll]ender`, paths `orchestrator/src/orchestrator/slot_lease.py`
- `slot-ended-signal` → built here; a per-task signal set at the release of a lease (open question 3). **PASS**.
  - delivered_check: manual. Where it lives and its name are decided in T3 (section 13, question 3).
- `permits-conserved` → built here; every permit released once on every exit in either order (invariant 4). **PASS**.
  - delivered_check: manual. Proved by the harness-loop tests over the five exits, not by a name.

---

## T4: The task picture and the claim-time check

*Intermediate; unlocks T5a. Serialized after T1 by scheduler.py.*

- `five-writing-steps` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_backfill_dep_status` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._apply_external_dep_policy` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._compute_delivered_check_cache` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_override_snapshot_gc` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._compute_effective_priorities` the steps whose read parts are split out (section 5.4). **PASS**.
- `ready-list-and-claims` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._eligible_for_dispatch` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_select_scored` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._dispatch_pin` wired: the started-set is checked at the ready list and added to at the ranked and pinned claims. **PASS**.
- `task-picture` → built here; a picture built only by read parts and kept for picture_max_age_secs (section 5.4). **PASS**.
  - delivered_check: grep present, pattern `[Pp]icture`, paths `orchestrator/src/orchestrator/scheduler.py`
- `claim-refuses-a-started-task` → built here; the ranked and pinned claims refuse a task already started (invariant 11). **PASS**.
  - delivered_check: manual. The started-set test already exists at the ready list, so a grep cannot tell the new claim-time check from it.
- `picture-writes-nothing` → built here; test 35 and an unchanged ordinary pass on a recorded pass. **PASS**.
  - delivered_check: manual. A behavioural property (invariant 12).
- `picture-event-loop-time-reported` → built here; the measured event-loop time of one picture in the task's result. **PASS**.
  - delivered_check: manual. A figure in the task's result, not code.

---

## T5a: The early-start selection in the scheduler

*Intermediate; unlocks T5b. After T1 and T4 (scheduler.py).*

- `delivered-checks-at-a-commit` → `grep:orchestrator/src/orchestrator/delivered_checks.py::run_delivered_check` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._compute_delivered_check_cache` wired: run_delivered_check takes a ref, read by the grep and path kinds. **PASS**.
- `system-holds` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_psi_gate` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.is_paused` `grep:orchestrator/src/orchestrator/harness.py::Harness._run_review_checkpoint` `grep:orchestrator/src/orchestrator/harness.py::Harness._probe_warm_base_health` `grep:orchestrator/src/orchestrator/harness.py::Harness.get_merge_halt_status` the host-load gate, pause, review checkpoint, warm-base health and merge halt the selection must honour. **PASS**.
- `fairness-bookkeeping` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._settle_fairness_on_dispatch` `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._bump_skip_and_maybe_park` the counter an early start must not bump. **PASS**.
- `eligibility-predicate` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler._eligible_for_dispatch` `grep:orchestrator/src/orchestrator/harness.py::Harness._enforce_cost_ceilings` the one predicate ranked and pinned starts use, and the cost-ceiling check the main loop runs before each dispatch. **PASS**.
- `lock-rule-and-picture` → producer:T1 producer:T4 upstream: the successor relation, the early-start dependency answer, the picture and the claim-time check. **PASS**.
- `tier-condition-refusal` → built here; the higher_tier_waiting refusal naming the task (ruling 3). **PASS**.
  - delivered_check: grep present, pattern `higher_tier_waiting`, paths `orchestrator/src/orchestrator/`
- `one-dependent-per-predecessor` → built here; a second dependent of one predecessor is refused predecessor_has_dependent (test 46). **PASS**.
  - delivered_check: grep present, pattern `predecessor_has_dependent`, paths `orchestrator/src/orchestrator/`
- `claim-in-one-step` → built here; request still live, file rule, tier condition, keys, relation and started mark in one uninterrupted step, with a release for a failed launch. **PASS**.
  - delivered_check: manual. Behavioural (tests 33, 34); no name marks it.

---

## T5b: The early-start coroutine and launch in the harness

*Intermediate; unlocks T6a. After T5a, T2a and T3.*

- `live-merge-requests` → `grep:orchestrator/src/orchestrator/merge_lane/types.py::InFlightMergeRegistry` `grep:orchestrator/src/orchestrator/harness.py::Harness.__init__` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._submit_to_merge_queue` wired: the harness owns the registry of live requests; the workflow awaits its request inside the slot. **PASS**.
- `event-vocabulary` → `grep:orchestrator/src/orchestrator/event_store.py::EventType` new members join this enum. **PASS**.
- `selection-stacking-borrowing` → producer:T5a producer:T2a producer:T3 upstream: the selection entry, the stack input and re-stack of create_worktree, the borrowing lease. **PASS**.
- `selection-and-dispatched-events-declared` → built here; early_start_selection and early_start_dispatched join EventType. **PASS**.
  - delivered_check: grep present, pattern `early_start_dispatched`, paths `orchestrator/src/orchestrator/event_store.py`
- `early-start-dispatched-emitted` → built here; the launch emits early_start_dispatched naming dependent, predecessor, stacked base, co-held keys. **PASS**.
  - delivered_check: grep present, pattern `EventType\.early_start_dispatched`, paths `orchestrator/src/orchestrator/`
- `early-start-selection-emitted` → built here; one selection event per change of outcome, at most once per waiting predecessor per interval. **PASS**.
  - delivered_check: grep present, pattern `EventType\.early_start_selection`, paths `orchestrator/src/orchestrator/`
- `trial-replay-refusal` → built here; step (b) runs the trial replay for a dependent with a stacked branch, refuses restack_conflict and remembers it per pair of tips. **PASS**.
  - delivered_check: grep present, pattern `restack_conflict`, paths `orchestrator/src/orchestrator/`
- `selection-coroutine-timer` → built here; the harness coroutine runs on submit and every pass_interval_secs while a predecessor has no dependent. **PASS**.
  - delivered_check: grep present, pattern `\.pass_interval_secs`, paths `orchestrator/src/orchestrator/`
- `early-start-mark` → built here; the mark on the dependent naming its predecessor. **PASS**.
  - delivered_check: manual. The metadata key is the architect's; tests 42 and 43 build and read it.

---

## T6a: Stop when the predecessor's work does not land

*Intermediate; unlocks T6b. After T5b and T2b.*

- `re-pend-and-cancel-paths` → `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._repend_for_requeue` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._exit_requeued` `grep:orchestrator/src/orchestrator/harness.py::Harness.cancel_workflow` `grep:orchestrator/src/orchestrator/harness.py::Harness.hard_cancel_workflow` wired: _exit_requeued writes pending through _repend_for_requeue. **PASS**.
- `requeue-cooldown-and-history` → `grep:orchestrator/src/orchestrator/scheduler.py::Scheduler.release` the requeued release arms the cooldown and records the requeue the stop must not count. **PASS**.
- `landed-question-and-slot-ended` → producer:T2b producer:T3 producer:T5b upstream: the answer at slot end, the signal to await, a live early start. **PASS**.
- `early-start-stopped-emitted` → built here; early_start_stopped with dependent, predecessor, cause, time worked. **PASS**.
  - delivered_check: grep present, pattern `EventType\.early_start_stopped`, paths `orchestrator/src/orchestrator/`
- `self-pause` → built here; early_start_paused and one escalation after stop_streak_pause stops in a row. **PASS**.
  - delivered_check: grep present, pattern `EventType\.early_start_paused`, paths `orchestrator/src/orchestrator/`
- `stop-streak-read` → built here; the stop streak compared with early_start.stop_streak_pause. **PASS**.
  - delivered_check: grep present, pattern `\.stop_streak_pause`, paths `orchestrator/src/orchestrator/`
- `hard-cancel-after-timeout` → built here; hard cancel after early_start.stop_hard_cancel_secs, pending written before the owner stamp is cleared. **PASS**.
  - delivered_check: grep present, pattern `\.stop_hard_cancel_secs`, paths `orchestrator/src/orchestrator/`
- `stop-not-counted-and-shutdown-excluded` → built here; the stop is a requeue that does not count; a shutdown emits and counts nothing. **PASS**.
  - delivered_check: manual. Behavioural (tests 22, 31); no name marks it.
- `work-in-progress-committed-at-stop` → built here; the stop commits the dependent's uncommitted work before the workflow exits, and a warm worktree is released only after that commit (test 48). **PASS**.
  - delivered_check: manual. Behavioural; test 48 proves it.
- `unknown-at-slot-end-re-asked` → built here; an unknown answer at slot end is asked again on the timer, with one escalation after unknown_streak_alert in a row (test 18 (i)). **PASS**.
  - delivered_check: manual. Behavioural; the alert key is checked at T6c.

---

## T6b: Settle when it lands

*Intermediate; unlocks T6c. After T6a.*

- `merge-phase-order` → `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._run_merge_phase` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._stamp_merge_phase_entered` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._handle_merge_gate_escalations` `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._rebase_before_enqueue` wired: _run_merge_phase stamps, then handles gating escalations, then rebases and submits; the settle goes between the stamp and the gating step. **PASS**.
- `merge-conflict-agent` → `grep:orchestrator/src/orchestrator/workflow.py::TaskWorkflow._resolve_and_resubmit` `grep:orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_merger_prompt` wired: the workflow invokes MERGER with this brief. **PASS**.
- `landing-events` → `grep:orchestrator/src/orchestrator/merge_lane/intake.py::enqueue_merge_request` `grep:orchestrator/src/orchestrator/event_store.py::EventType` merge_finalized carries state (done, already_merged and others), emitted from enqueue_merge_request's _on_finalized. **PASS**.
- `landed-question-replay-slot-ended-stop` → producer:T2b producer:T3 producer:T6a upstream. **PASS**.
- `early-start-settled-emitted` → built here; early_start_settled with the answer, replayed, conflict resolved, verify re-run. **PASS**.
  - delivered_check: grep present, pattern `EventType\.early_start_settled`, paths `orchestrator/src/orchestrator/`
- `merge-entry-wait-alert` → built here; the wait at merge entry and one escalation after early_start.merge_entry_alert_secs. **PASS**.
  - delivered_check: grep present, pattern `\.merge_entry_alert_secs`, paths `orchestrator/src/orchestrator/`
- `paused-replay-brief` → built here; the merge-conflict agent's brief for a paused replay: never merge main in, never rebase the whole branch. **PASS**.
  - delivered_check: manual. Brief wording is the architect's; test 17 proves it.
- `verify-rule-after-landing` → built here; one more task verify unless the landing is a merge commit whose second parent is the stacked base (choice 5). **PASS**.
  - delivered_check: manual. Behavioural (tests 13 to 16).

---

## T6c: The landed question at every later start

*Intermediate; unlocks T7. Depends on T6b.*

- `re-cut-gate-in-run-slot` → `grep:orchestrator/src/orchestrator/harness.py::Harness._run_slot` `grep:orchestrator/src/orchestrator/harness.py::Harness._run_recut_ruling_gate` wired: _run_slot runs the re-cut gate before the worktree is touched. **PASS**.
- `issuing-a-ruling` → `grep:orchestrator/src/orchestrator/recut_ruling.py::issue_ruling` `grep:orchestrator/src/orchestrator/harness.py::Harness._issue_restart_recut_ruling` wired: the restart resolution already issues a ruling from code through issue_ruling; the automatic issuer follows that shape. **PASS**.
- `requeue-replay-path` → `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._reuse_warm_lane` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps._rebase_requeued_onto_main` `grep:orchestrator/src/orchestrator/git_ops.py::GitOps.unstack_from_unlanded_base` wired: the existing requeue path that replays onto main on landed. **PASS**.
- `recorded-landings-after-restart` → `grep:orchestrator/src/orchestrator/event_store.py::_MERGE_FINALIZED_ROWS` the event-store selector for merge_finalized rows. **PASS**.
- `landed-question-and-settle` → producer:T2b producer:T6b upstream. **PASS**.
- `early-start-recut-emitted` → built here; early_start_recut with dependent, predecessor, ruling. **PASS**.
  - delivered_check: grep present, pattern `EventType\.early_start_recut`, paths `orchestrator/src/orchestrator/`
- `unknown-streak-alert` → built here; one escalation after early_start.unknown_streak_alert unknown answers in a row. **PASS**.
  - delivered_check: manual. T6a already reads this key at slot end, so a use-site grep would be green before T6c lands; test 38 proves the start-time alert.
- `automatic-issuer-named` → built here; recut_ruling.py names the automatic issuer (early start: predecessor's work never landed). **PASS**.
  - delivered_check: grep present, pattern `early.start|[Pp]redecessor`, paths `orchestrator/src/orchestrator/recut_ruling.py`
- `re-cut-drops-record-and-mark-once` → built here; the re-cut drops the ledger record and the mark; never issued twice for the same predecessor and base. **PASS**.
  - delivered_check: manual. Behavioural (tests 25, 37).

---

## T7: Operator views, and the default turned on

*Intermediate; unlocks T8, T9. Depends on T6c. Soft pre-condition: task 10211.*

- `slot-count-alarm` → `grep:dashboard/src/dashboard/data/burndown.py::compute_parity_alarm` `grep:dashboard/src/dashboard/data/burndown.py::_row_for_root` `grep:dashboard/src/dashboard/data/redux_api.py::_project_block` wired: _project_block serves compute_parity_alarm over the persisted in_progress_live column; _row_for_root logs the same comparison. **PASS**.
- `runtime-state` → `grep:orchestrator/src/orchestrator/harness.py::Harness.task_runtime_snapshot` `grep:orchestrator/src/orchestrator/task_runtime.py::TaskRuntimeState` `grep:shared/src/shared/task_runtime_state.py::TaskRuntimeEntry` `grep:escalation/src/escalation/server.py` get_task_runtime_state projects the harness snapshot onto TaskRuntimeEntry. **PASS**.
- `operations-doc` → `grep:OPERATIONS.md` a section on early starts. **PASS**.
- `borrowed-slot-lender` → producer:T3 producer:T5b upstream. **PASS**.
- `alarm-allows-borrowed-slots` → built here; each burndown snapshot row records the number of live early starts and the slot-count alarm adds it to the cap. **PASS**.
  - delivered_check: grep present, pattern `early_start|[Bb]orrowed|[Ll]ender`, paths `dashboard/src/dashboard/data/`
- `runtime-state-names-lender` → built here; the runtime entry of an early-started dependent names the task whose slot it borrows. **PASS**.
  - delivered_check: grep present, pattern `[Ll]ender|early_start|[Bb]orrow`, paths `orchestrator/src/orchestrator/task_runtime.py`, `shared/src/shared/task_runtime_state.py`
- `enabled-by-default` → built here; early_start.enabled defaults to on (choice 7, test 41). **PASS**.
  - delivered_check: manual. A default value; a grep for its literal text is too brittle to gate T8 and T9 on.

---

## T8: Gate: read the first real pairs

*Signal leaf. Pure gate, delayed two days after T7 is done.*

- `events-to-read` → `grep:orchestrator/src/orchestrator/event_store.py::EventType` producer:T1 producer:T5b producer:T6b upstream: lock_acquired with successor_of, early_start_dispatched, early_start_settled, and the existing merge_queued and merge_finalized. **PASS**.
- `end-to-end-sequence-for-one-real-pair` → built here; the ordered sequence of section 9, T8, read from the event store for one real pair. **PASS**.
  - delivered_check: manual. A reading of production events after two days; nothing in the tree changes.

---

## T9: Gate: read two weeks of early starts

*Signal leaf. Pure gate, delayed fourteen days after T7 is done; carries the end-to-end proof if T8 closed without a completed pair.*

- `counts-to-read` → producer:T5b producer:T6a producer:T6b producer:T6c upstream: refusals, stops by cause, settles, re-cuts. **PASS**.
- `two-week-report-and-stage-2-recommendation` → built here; counts per week against section 2, and a recommendation on stage 2. **PASS**.
  - delivered_check: manual. A report for the operator.
