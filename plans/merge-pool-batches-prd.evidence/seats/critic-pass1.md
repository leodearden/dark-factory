# Critic pass 1: `/tmp/gate-10069/draft/merge-pool-batches-prd.md`

Read-only against main `7c93125c63` (contains `09bd5bd7b1`). Scratch is under `/tmp/gate-10069/critic1/` (`infer.py`, `pid/r1`, `pid/r2`). Code paths are relative to `/home/leo/src/dark-factory/orchestrator/src/orchestrator/` unless absolute.

## A. Answer-changing defects

### Against the code

**A1. §6.1 "`built_on` is the tree of the in-flight entry whose successor base equals this item's base … empty when the item was built on main."** Three reachable cases are undefined:
- **Base landed but is not main's tip.** A speculative item is never re-merged for a moved main (`merge_lane/verify_dispatch.py::VerifyDispatcher._stale_merge_reason`), so this occurs whenever the entry ahead finalizes before the dispatch.
- **Base is an unlanded commit that no live entry owns.** This is task 8933's class, 2–3 a day in its specimens. `inflight_verify.py::InflightVerifyRun._settle_chain` requeues without recording the commit dead.
- **The owning entry is vacated.**

A naive build gives an empty `built_on`, so a red reads as "red on main" at 0.94.
*Requirement:* define members and `main_sha` for each case; an unresolvable base writes a no-evidence row, never an empty `built_on`. §5's 8933 row ("none") must become a dependency or a stated interim rule.

**A2. §4 "One dispatch path for every verify" and §6.2 "one row for every verify that returns: batch, single, gate re-verify, train."** These verifies bypass `_place`/`_start_verify` and have no `InflightEntry`:
- trains, inline in the merger (`merge_lane/worker.py::_do_train_merge` via `_leased_train_verify`);
- the train's gate re-verify (`merge_lane/gates.py`, `dispatch_reverify=False`);
- `worker.py::reverify_member_solo`, called from `workflow.py`;
- `VerifyDispatcher.gate_history_leg`, which is not a full gate.

*Requirement:* per path, say row or no row, who writes it, and what a `TreeMember` is for a `GroupMergeRequest` (N tasks, one tip).

**A3. §6.2 "`RED` is a code verdict as `_is_code_verdict` defines it" contradicts `NoEvidence{TIMED_OUT, INFRA, NOT_FULL_GATE}`.**
- `worker.py::_is_code_verdict` is a module function: `not out.verify_skipped`. Timeouts, transient-infra, flock contention and trivial-pass-withheld are all "code verdicts" by it.
- `verify_failure_outcome.py::_transient_infra_outcome` carries no structured marker, only a reason prefix.
- A `VerifyResult.trivial` pass ran no suite, yet would be `PASSED` and clear every key in its tree.

*Requirement:* a total mapping from outcome facts to `Verdict`/`NoEvidence`, with the structured carrier α adds (`verify_failure_outcome.py` is missing from α's files). A trivial pass clears nothing.

**A4. §6.4 "These replace every read of `_next_dispatch_unchained`, which is removed with its port"; row 22 "no suspect".** The mark has a second job the draft drops.
- `InflightVerifyRun._on_verify_error` marks the owner so a raising batch cannot re-form (guard 2b, "LOOP GUARD").
- That is pinned by `test_merge_queue_deep_landing.py::test_chain_arm_exception_stays_non_adopting` and `test_merge_queue_deep_unchained_mark.py::TestBAChainArmErrorStillTerminates`.
- With VERIFY_ERROR/TIMED_OUT/INFRA as no-evidence, nothing stops the same batch re-forming.
- A batch timeout also bumps the owner's counter (`RedVerifyContext.bump_timeouts`), so the owner is abandoned for a link's hang.

*Requirement:* what bounds repetition for each no-evidence class, and its restart status under L8.

**A5. §6.3 I1 against §6.2 "written … before the batch's links are released."**
- `InflightVerifyRun.run`'s `finally` ends the reservation and then awaits, before `_settle_chain`.
- The record can go in `_take_verdict`, but the posterior changes only when the off-thread recompute lands.
- A concurrent `chain_snapshot` or pick reads the old cache. This is task 10624 item 1's window, measured there at 3–57 s.
- A conviction can likewise fire from a posterior older than a recorded void (C3).

*Requirement:* a versioned posterior; `is_suspect` conservative for keys in observations newer than the cache; conviction only at cache version = ledger version; a late recompute must not overwrite a newer one.

**A6. §6.4 "A suspect whose group has a member in an unreturned verify is passed over."**
- The merger picks ahead of dispatch. A picked suspect sits in `_verifier_queue` or the redispatch park (MERGING, AWAITING_VERIFY, REDISPATCH_PARKED, DISPATCHING) and is in no verify.
- Cascade re-merges (`_remerge_torn_down_entry` → `request_redispatch`) never pass the pick.

*Requirement:* the predicate over lifecycle states from pick to verdict, and the rule for parked items.

**A7. D2 / §6.1, content id.** Scratch results (git 2.43):
- **"For a fixed tip the merge base does not move" is false.** In `pid/r2`, tip `470aa8d` has id `7693bc7c…` while the frozen tip under it is unlanded and `25ebb906…` after it lands (merge base `ff6aa46` → `4eca840`). The lane does this itself: `suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items` rebases waiting requests onto the unlanded frozen tip. Stacked branches behave the same (`GitOps.unlanded_stack_base`, `own_delta_upstream`).
- **Row 11 (unstack keeps identity) fails as specified.** `merge-base(main, ref)..ref` includes the stack base's changes before the unstack and not after.
- **A conflict-resolved rebase with the same intent changes the id.** `pid/r1`: `71c3d1f4…` → `8b7ec5f4…`, because the hunks coalesce.
- **Batch links.** They are merged by `worker.py::build_chain` → `GitOps.merge_branch_into_worktree`, never by `classify_and_merge`. "Computed at enqueue, recovery and each merge attempt" gives a link an enqueue-time id beside a build-time tip.
- **Nothing refreshes a waiting request whose branch moved** (row 10).
- **A second definition of "same content".** `merge_lane/content_relation.py::classify_content` and `landing_evidence.py::range_patch_ids` are on main (task 10215, done). §4's "no patch-id helper" row omits them (heuristic 11).

*Requirement:* identity over the branch's own delta, excluding commits reachable from main, from in-flight commits and from a recorded stack base. Memoise on (tip, base). Compute a member's id from its tip in that tree. State the relation to `ContentRelation`. A request with no id is never convicted.

**A8. §6.6 "voids every … observation whose `main_sha` is M."**
- The probe tests main HEAD at probe time and emits only if `_main_is_still_at` (`main_health_probe.py::_run_deferred_main_health_probe`).
- For a stacked tree whose base landed during the verify, M is that base's merge commit, not the tree's `main_sha`. The triggering observation is never voided.
- There is no `main_health_red` event type. It is a `merge_attempt` row with `outcome=main_health_red`, with two emit sites: `_run_deferred_main_health_probe`, and `classify_main_health_red` on the sync path.
- For cargo projects and lint/type reds both id lists are empty, so "all of them" voids every red on M. The observation has no (category, cause_hint) to match the probe's signature path.

*Requirement:* containment (M is `main_sha` or a member's merge commit or chain tip). Put the failure signature on `Observation` in α, since rows cannot be backfilled. Name the source of `failing_ids` (`verify.py::known_failing_ids_on_main` is the cache peek).

### The model

**A9. §6.3 Rates.**
- "Blocked on its own red verify" is not a row and `note_gone` carries no reason.
- "Dropped when none of its keys belongs to a live request" contradicts "last 200 decided keys": decided keys are gone by definition.
- `false_failure` has no window. One undetected 30-red episode needs about 1,700 sound-tree verdicts to return under 0.018.
- Convicted keys are neither cleared nor own-red, so the defect rate is biased low once ON.
- An unexplained red needs every member cleared. Conviction, or a content change after a block, prevents that, so `false_failure` is biased low.
- The live ledger and a rebuild read different windows, so I4 and S3 fail.

*Requirement:* each rate as a pure function of a stated row window, with rules for convicted, void, gone and cleared-and-own-red keys.

**A10. D5 "`p_bad` > 1.2 × defect rate".** A red tree of n members makes no suspect once (1−d)ⁿ(1−f) < 1/6: about 12 members at d=0.14, 8 at 0.20, 7 at 0.24 (computed with `infer.py`).
- A batch of 12 on main at d=0.14 gives 0.167 against a threshold of 0.168.
- `MergeDeepConfig.depth_table` reaches 12, and `chain_policy.choose_depth` returns row 0 with no estimate.
- Sizing reads a 30-verdict window; the threshold reads 200 keys.
- Repeating the identical red does not raise `p_bad`, so the batch re-forms.
- At cold start (d = f = 0.5) no red tree of two or more makes a suspect (0.533 < 0.6).
- S1 cannot hold after dispatch.

*Requirement:* isolation of a red tree's members on a sound base that does not depend on size or the fitted rates, or a stated bound.

**A11. §6.3 "A group over `max_group`".**
- One tree exceeds 16 in ordinary operation: a batch of 6 on two entries of 6 is 18 keys (three entries in flight; `cond_report.out` shows k6 batches on 6-member bases).
- `infer(...)` takes no order argument, and in-flight or gone keys have no pool order.
- Removing members from a red observation without marginalising them inflates the others. The sixteenth can be convicted when the dropped member was the culprit.

*Requirement:* a bound that leaves the likelihood unchanged (keys with identical observation membership are exchangeable), and no conviction from a truncated group.

**A12. §2 "An undetected red main cannot make the lane convict more."** False for the last member.
- {A,B,C} goes red on a red main; main heals; A and B pass; C is convicted at 0.94 before "all members pass later" can occur.
- Shadow truths taken during a red main are all red, so the switch turns ON during the episode.
- A pass clears for ever. Under task 10721 (pending) a head that already passed is re-verified in a batch on a moved main, and its skew red falls on the links.

*Requirement:* restate the premise with its real bound; a void observation is no truth.

### The switch

**A13. §6.5 State "a pure function of the recorded rows".**
- No row records a truth or a state change. "Rows after the suspension" has no marker.
- With `rebuild_row_cap` truncation, replay restarts in SHADOW.
- "10 shadow rows … at most 1 pass" does not say first, last or cumulative.
- SUSPENDED re-arms itself, so the state flaps, one L1 per flap.
- Binomial figures (mine): P(ON) at true precision 0.9 / 0.8 / 0.7 is 0.74 / 0.38 / 0.15. P(5 or more of 20) at wrong-rate 0.1 / 0.2 / 0.3 is 0.04 / 0.37 / 0.76.
- With one control in five, a collapse to all-wrong delivers about 20 wrong convictions before five control passes.

*Requirement:* state-transition rows; truth as a defined join over `merge_evidence` rows; window semantics; hysteresis or an operator re-arm.

**A14. §6.5 threshold, §2 "within 2%", δ's signal.**
- 0.9 needs a false-failure estimate at or below d/(9(1−d)): 1.8% at d=0.14, 0.97% at 0.08. One unexplained red in 200 switches conviction off at 0.08.
- `/tmp/gate-10069/sim/results/final_tables.txt` at q=0.08 (W1, main-red on): Jc-J1d makes 0.2 right / 0.01 wrong per 100 landings and lands 1.973 per verify-hour. TR lands 2.041 (3.3% more); threshold 0.8 also gives 2.041.

*Requirement:* say when δ fires; show "cannot fire, and why" in the snapshot; give δ a signal producible when no conviction occurs.

### Restart, deadline, outcome path

**A15. §6.4 Restart, S3, goal 5.**
- The truncation test reads first-enqueue time, which recovery loses (task 5623, pending; `merge_lane/queue_store.py::PersistedMergeRequest` has no such field). It never fires.
- `EventStore.fetch_recent_events_by_type` returns `[]` on error, and `from_event_data` raises on an unknown enum. The result is a silently empty ledger or a dead `run()`.
- The suspect-since clock and the every-fifth counter are pick state under L8, with no restore rule.
- "The next pick is the same" cannot be observed: in-flight requests are waiting after a restart.
- The confirmed-red (M, F) needed for "voided as it is recorded" is in no rebuild row.

*Requirement:* each item restored from named rows or listed as dropped; a rebuild row on every start with counts; per-row skip with a threshold.

**A16. §6.4 Deadline and S2.**
- "Picked ahead of every normal-lane request" is an order rule. §5 gives order to task 9935's `pool_order()` and says "suspects are ordered like any request".
- S2 is false under a halted normal lane (`_auto_heal_main_health_deferred` → `halt_lane('normal')`), with no free host, or after a cascade discard.
- The default is "printed by ε, committed by β2", but β2 does not depend on ε. It would also bake dark_factory's p95 into shared code.

*Requirement:* one owner for the promotion; S2 with its exclusions; a default the lane derives per project, or a real dependency.

**A17. §6.5 What.**
- `MergeBlockCategory` values must be `EscalationCategory` members (`merge_lane/outcome_facts.py`), so δ needs a new category in `/home/leo/src/dark-factory/escalation/src/escalation/categories.py`. That file is not in δ's files.
- A blocked outcome has three consumers: `TaskWorkflow._submit_to_merge_queue`, `recovered_merge_owner.py` (reads `outcome.block_category`), and MCP `merge_request` callers with no workflow.
- `_mark_blocked`'s arguments (steward or human, dry-run spawn) are unstated.
- The agent receives test ids only. Observations store no failure output.
- Thrash: the merge-phase loop in `workflow.py` runs `_check_merge_outcome_thrash` whenever `_last_merge_block_reason is not None`. That field is left over from an earlier generic block, so a conviction arm that does not reset it advances the counter (row 7). The existing precedent is a never-repeating reason (`SuffixConflictTracker._escalate`).

*Requirement:* all three consumers, the routing arguments, the stored report, and the thrash mechanism named.

### Gates, sizing

**A18. §5 (G4).**
- **Not listed, same symbols:**
  - `plans/one-merge-request-per-task-prd.md` (tasks 10215 done, 10216, gate 10217): R2 updates a waiting request's content under one request id; links become one record per link.
  - Task 7824: the chain is built inside the verify task, so "tree set once at dispatch" fails.
  - Tasks 9171 and 9156 move γ's sites.
  - Task 9182 changes `_retire_item`, the `note_gone` site.
  - Task 9207 deletes `_last_merge_*`.
  - Task 9766 moves `EventType`.
  - Tasks 10136, 8329 and 9133 also edit these sites.
- **Wrong:**
  - Cancelling 10624 leaves 10759 depending on it.
  - 10759's contract hard-codes the mark (D5–D16, rows H3/H4/H19, `MemberExit` has no "released"). `plans/merge-chain-visibility-prd.md` and its manifest need a revision, not a note, and no task owns it.
  - "Companion text in section 8" does not exist.
  - Task 7921 adds `probe_sha` and F to the outcome while γ adds them to the row: a "whichever lands second" seam.

**A19. §8 θ.**
- No contract, no boundary rows, no store, format or staleness rule.
- "A quarantined host is still quarantined" reverses a recorded decision (`merge_lane/host_quarantine.py`: "Rely on that coupling rather than persisting the set").
- `harness.py::Harness._rehydrate_merge_halt` already restores three halt categories.
- "Parked for redispatch is still first" depends on task 5623.

**A20. §8 sizing and order.**
- **α:** 12 listed source files, plus `verify_failure_outcome.py`, `merge_pool_facts.py` and five or more test files. That is over the 15-file review trigger and covers four concerns: identity, tree, outcome facts, row.
- **β2:** 14 rows and the mark removal, plus mark assertions it does not list in `test_merge_queue_deep_dispatch.py`, `test_merge_queue_deep_landing.py`, `test_merge_queue_deep_integration_gate.py`, `test_merge_lane_verify_dispatch.py` and `_merge_deep_scene.py`.
- **Order:** γ need not precede β2. §2's own premise names δ as the task that needs γ. α → β1 → β2 → γ → δ lands the throughput fix one task sooner.

## B. Minor

1. I3 is vacuous: a key sound in every consistent hypothesis has `p_bad` 0 by normalisation. The intended invariant is "1 only if defective in every consistent hypothesis".
2. `failing_units`, `VoidReason` and the order behind "first 50" are undefined. `rebuild_row_cap` has no default and no per-type-or-total rule. `max_group` is 16 only in prose.
3. Symbol slips: `_is_code_verdict` is not a method; `queue_store.py` is `merge_lane/queue_store.py`; the main-health emit site is `_run_deferred_main_health_probe`, not `spawn_main_health_probe`.
4. R10 not implemented: 10/1, 5 of 20, every fifth, ratio 1.2 and 16 have no measured basis stated.
5. G1: `merge_queued.content_id`, `flake_suppressed` and `merge_evidence_rebuild` name no consumer.
6. L7: ε prints no verifies per landing, so §2's gains cannot be checked on the lane.
7. "Train forming skips suspects": declared trains are formed in the workflow, and coalescing is off in every yaml. Name `_default_coalesce_exclusion_reason`.
8. C1 gap: a link whose reservation expired (`chain_reservation_timeout_secs` 7200) or was cancelled is waiting and unreserved while its tree is unreturned. `_land_chain_prefix` then stops at `take_buffered` returning `None`.
9. Tree-keyed fakes are private to the test file β2 rewrites. Hoist them in α.
10. δ's "dashboard outcome colours": the colour-family table is task 9852's (pending). Main has only `outcome_colors.py::assign_outcome_colors`.
11. INV-1: no validators on thresholds. INV-4: the id-`None` count, group-cap hits and no-evidence streaks have no threshold.
12. A voided red's members re-form on the still-red main. Say so, and say what bounds it.
13. Snapshot: no "passed over since / deadline at".
14. Phase 2 would force redoing D5 (A10), `max_group` (A11) and one-request-one-round (`ChainReservations._refuse_foreign_hold`), which sub-batches break.

## Earlier defects not answered

- **Pass 1**
  - A1: pick-ahead (my A6).
  - A6.5: halted lane (A16).
  - A14 class: cancelled prerequisite (A18).
  - A15: "whichever lands second", for 7921 (A18).
- **Pass 2**
  - A8.2: bounce rebase (A7).
  - A8: "no id, never convicted" (A7).
  - A10: triggering failure not voided (A8).
  - A11: stored report, half-answered (A17).
  - B10: when inference runs (A5).
  - B15: counters without thresholds (B11).

## Evidence spot-checks

- **"0 of 114, at most 2.6%".** 114 is the green count. `ffr2.py` counts a red as a false failure only if every member is later cleared with the same content. Reds whose members changed content are excluded, so this bounds the estimator, not the rate.
- **"43 of 43".** This is the extended truth. Strict truth is 21 of 21 with 24 unknown. The extension rests on two hand-inferred contents (`cond_build.py::INFER`: tasks 7736 and 10137), and the reds come from 7–9 distinct defective contents.
- **"25 … 18 … 0.07 to 0.09 against 0.09 to 0.12".** Holds (`cond_report.out` sections 1 and 3).
- **"0.21 against 0.30".** Holds. "Within 2%" holds at q=0.14 only (A14).
- **"+7 to +17%" and "+11 to +18%".** The grid gives +6 to +17% with main-red on (+1 to +5% off) and +8 to +17%. Of step 1, the unlanded-work part alone is +1.3 ±1.3% at q=0.14.

## Leo's call

1. **Threshold 0.9 with the L4 estimator** leaves conviction idle at defect rates near 0.08. Options: keep 0.9; use 0.8 (1.5 right / 0.19 wrong per 100 in the sim); or defer δ until rows show the rate.
2. **Switch constants and re-arm.** Options: automatic as drafted (flaps; about 20 wrong convictions on a collapse); operator re-arm after a suspension; or a higher control share.
3. **θ's host quarantine.** Restore it across restarts, or keep the recorded decision and list it as dropped.
4. **Which requests can be convicted.** Workflow-awaited only, or also recovered and MCP-submitted ones.
5. **Delaying pinned 10995 behind α and 9935 behind β2**, or reordering as in A20.

## Checked and sound

- The mark is set in `_settle_chain` and `_on_verify_error`, read in `chain_snapshot` and guard 2b, and cleared only in `_retire_item`.
- The worked values in §6.3 reproduce: 0.378, 0.234, 0.378, 0.142, 0.942.
- The 54 and 102 sound-tree-verdict figures reproduce.
- The trees give the right answer for a single red later discarded by the cascade, and for `_void_and_remerge`.
- Row 13 (a gone member blocks elimination) gives 0.52.
- The pair-fails-only-together case resolves through the first landing.
- `-U0` ids survive a clean rebase; empty and `.task/`-only diffs give no id; binary and mode-only changes are distinguished.
- `MergeIntake._clique_minimal_visible`, `remove_buffered`, `SuffixConflictTracker._escalate` and `ChainReservations.holds` are as stated.
- Recovery (startup step 1c0a) runs before `_launch_merge_worker_loop`, so recovered requests exist at `run()`.
- `fetch_recent_events_by_type` spans runs, oldest first.
- Ratchet ceilings (1,500 lines, cognitive 15) and the `--authorize-raise` path are as stated.
- `enqueue_merge_request` is async.
- A link's tip is its merge commit's second parent (`--no-ff`).

## Not checked

- Counts I did not recompute from `ev.jsonl`: 13 of 58, 30 of 62, 31 restart-crossers, 1,252 of 1,300, 0.14 over 1,237.
- The simulator's code (`lane.py`): whether it fits rates online or models the cap.
- `_merger_loop` and `_look_ahead` bodies line by line.
- `snapshot()` / `get_merge_queue` plumbing and the dashboard.
- Finalized `plan.json` files of the seam tasks.
- Tasks 8532, 8932, 5623 and 7922 beyond their titles.
- Reify's runner facts.
- I ran no repository tests.
