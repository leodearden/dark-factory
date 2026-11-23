# Merge pool, part 2: find the culprit of a red batch by re-running what failed

**Status:** active. Authored 2026-11-20; decomposed the same day. Part 2 of 2; part 1 is `plans/priority-at-later-gates-prd.md`.
Authored for gate task 10069 in an attended session with Leo; his rulings are in section 3.
**Code anchors** verified against main `9856ed15dc` (2026-11-20). Main moves fast: cite by symbol; re-locate at
implementation time.
**Evidence:** `plans/merge-pool-batches-prd.evidence/`, written `evidence/` below. Its `REPORT.md` ties every
figure here to a seat report with its counting rule; the scripts, the simulator and the statistical design this
replaces in phase 1 (`draft3-statistical-design.md`) are beside it.

## 1. Goal

When a batch of merge requests fails its verify, the lane finds out which member broke it by re-running only what
failed, once for each member alone on main. A member whose own tree fails is sent back to its workflow with that
run's failure. The other members are treated as verified and ride a later batch. A failed batch then costs one
verify plus a few minutes, so batches can be several times larger than today.

Today a red batch makes every member verify alone (the un-chained mark, tasks 10617 and 10723): one full verify
per member, 26 to 42 minutes each, and the mark is lost on restart. Batch size comes from a table that assumes
that cost (task 8528).

Terms. A *batch* is what the code calls a deep chain: one dispatched request (the owner) plus further waiting
requests (links), merged one on top of the other onto a base and verified as one tree, the *tip*. The *root* is
the main commit under the batch; the *base* is the commit the owner was merged onto, which is the root or
unlanded work stacked on it. A member's *own tree* is the root with that member alone merged onto it. A *check*
is a failing test, or a lint or type-check leg of one module.

What an operator observes when it has landed:

1. A red batch is followed within minutes by a `merge_scan` row giving, for each failing check, its result on the
   root, on the base, on the tip and on each member's own tree.
2. A member whose own tree fails a check twice, where the root and the base pass it, is resolved `blocked` as a
   verify failure with that run's failure report, before the merger's next pick. It gets no verify of its own.
3. The other members are *cleared*: `get_merge_queue` shows them as such, and the rule batches them as low-risk
   members, up to 24 in a batch.
4. If the checks already fail on the root or on the base, no member is blamed.
5. A red batch the scan cannot settle sends each member to verify alone; once one of them is blocked, the others
   are batched again.
6. Batches are sized by one cost rule from fitted rates; `chain_depth_reason.basis` is `cost_rule` and the depth
   table no longer exists.
7. After an orchestrator restart the same requests are cleared, the same requests verify alone, and a scan that
   was cut off runs again.

## 2. Background and premise validity (G6)

All figures: dark_factory `runs.db`, syslog and git. The counting rule for each is in `evidence/REPORT.md`.

**Measured.**

- *Failures are deterministic.* Over 214 verdicts with the tested tree rebuilt from git (2026-11-15 to 11-20), no
  green tree contained a request proved defective with that content, and no red tree on a sound main had every
  request in it proved sound (4 undetermined). Eight test files explain all 58 red batches.
- *The failing tests of a red batch are usually runnable.* The flake gate re-runs a red verify's failing tests in
  isolation on the red tree. Of 104 red batches since 2026-10-21, 87 (84%) had a set that failed again in
  isolation, 2.1 tests on average. Of the other 17, 10 had test ids the flake gate could not map to a module and
  7 had a lint or type-check leg red as well (6 of those 7 with failing tests too).
- *The 10 unmapped ones are batches whose owner has no module.* Their verify is the global test command, which
  stops at the first failing module. Their ids map against the project's full module set, 10 of 10.
- *Most failing tests exist before the batch.* In 89 of 105 red tips every failing id is present on the base; in
  4 its file is absent; in 1 the test was added to an existing file.
- *Flaky tests are common.* 132 merge verifies in the same period were turned from red to green because their
  failing tests passed when re-run. One isolated failure is therefore not proof.
- *56 of the 104 red batches were verified on the laptop.* The laptop can only run a full verify.
- *Re-running a few tests on one tree is cheap.* An estimate, since nothing records it: preparing a worktree's
  environment takes 12 s at the median (43 s at the 90th percentile), a merge 1.3 s, and one module's failing
  tests 10 to 60 s. That is 0.5 to 2 minutes a tree, against a full verify of 26 minutes on the laptop and 42 on
  the workstation: 2 to 8% of a verify. A lint leg takes seconds; a type-check leg 11 to 220 s by module.
- *A red on unlanded work is often not the batch's fault.* 13 of 58 red batches were red only because of unlanded
  work ahead of them; today all their members are marked anyway.
- *The mark does not survive a restart.* After task 10617 deployed, all 31 red-batch members that were batched
  again before their solo verify had crossed an orchestrator restart.
- *Larger batches build.* A link fails to merge about 2% of the time at each of the first five positions (19 of
  999 link attempts, repeats of one item counted once); positions beyond the fifth are not measured. At that rate
  81% of 12-member builds complete, and one cut short keeps about 11. Build time does not grow measurably with
  size (4 to 9 s at the median for 2 to 6 members).
- *Verify time does not grow with batch size* over 2 to 6 members (slope −0.7 minutes per member, interval −2.6
  to +1.3). Larger sizes are not measured.
- *Landing is slow per member today.* 246 s per member after the first at the median; task 10998 removes most of
  it.
- *Per-request defect rate.* 0.14 over 1,237 requests (0.07 to 0.24 by week).

**Simulated** (the 8528 study's lane simulator, extended and re-validated; 2 verify hosts, 3 entries in the
frontier, 75 waiting, stacked as today; each member scanned alone, two failures to convict, the scan off the
verify hosts and one at a time at 5% of a verify per tree; the release rule of L9 as the fallback). Landings per
verify-hour:

| Defect rate | Today | No scan, the release rule alone | Scan, every red scannable | Scan, 16% of culprits not scannable | The same at 5% |
|---|---|---|---|---|---|
| 0.05 | 2.06 | 2.73 | 8.35 | 5.73 | 7.30 |
| 0.08 | 1.57 | 2.02 | 7.04 | 4.30 | 5.78 |
| 0.14 | 1.11 | 1.35 | 5.61 | 2.87 | 4.23 |
| 0.20 | 0.87 | 0.97 | 4.97 | 2.11 | 3.33 |

- These columns are at a verify of 43 minutes at the median. At today's 75 minutes, with 16% of culprits not
  scannable, the lane lands 6.4, 4.8, 3.2 and 2.4 requests an hour; today's lane lands 2.2, 1.7, 1.2 and 1.0.
  About 2.4 arrive.
- At 43 minutes, a defect rate of 0.14 and 16% not scannable, a defective request is sent back after 10 hours at
  the median (32 today) and a sound one lands after 10 (29 today). The scan blocks no sound request in the model.
- *What cannot be scanned costs more than anything else here.* Each 5 points of culprits that cannot be scanned
  cost 8 to 13% of throughput. The model is pessimistic: one such culprit makes its whole batch unscannable.
- *The fallback.* For a red batch that cannot be scanned, at a defect rate of 0.14: every member verifies alone,
  2.59; the same with the rest released once one member is blocked, 2.87; re-verifying in halves, 2.68; draft 3's
  statistical design repriced for the purpose, 3.16; draft 3's design unchanged, 0.37, because it re-batches the
  same members and waits for a scan that cannot settle them.
- *Concurrency.* Two member trees side by side give 5% more and hold a batch's members 0.4 hours at the median,
  against 1.4; four hold them 0.25. One scan at a time remains enough.
- *The price of a tree.* With trees run one at a time, scanning every red batch stops beating never scanning when
  one tree costs 0.43 to 0.57 of a verify, across three fallbacks.

**Premises checked.**

- *A scan conviction rests on the same evidence as a solo verify failure, restricted to the failing checks.* The
  member alone on main fails a check twice, in the isolated form the flake gate uses, and main passes it, on one
  host. A broken scan environment fails the root as well, so it cannot convict (N1). The result does not depend
  on unlanded work, which a solo verify's does.
- *"Cleared" is a probability, not a verdict.* A cleared member passed every failing check alone and sat in a tip
  that passed everything else. It has not passed the unscoped type-check gate, which only runs after the rest is
  green, and it may still fail together with another member (1 red batch in 58). It lands only inside a tree that
  passes a full verify.
- *The simulated lane is a model.* Its defects never interact, it has no merge conflicts, and its pool never runs
  dry. Three things in the contract differ from it and were not simulated: members are tested on the root where
  the model tested them on the base (the same result when defects do not interact, one more tree for a stacked
  batch, and a check that only the unlanded work brings ends in the fallback; how often is not measured); a
  cleared member of a batch that takes the fallback becomes a new request again, where the model kept it cleared
  (1 to 2% on the going cost by the rule's own arithmetic); and the size of a batch's first kind is computed
  directly, where the model added members one at a time with a look-ahead.
- *The rule is well defined over its range.* The reference computation (`evidence/scripts/rulecheck/rule9.py.txt`)
  converges in every one of 1,278 cells (defect rates 0.01 to 0.5, unscannable shares 0 to 1, scanning on and
  off), and the first kind's size is the cheapest per member in each.
- *The rates are estimates today.* The lane records what each needs from the first scan on, and the rule reads
  the recorded values.

## 3. Resolved design decisions

**Carried from part 1 and gate 10069 (not reopened).**

- **R1.** Nothing merged or being verified is displaced by a later arrival.
- **R7.** A request implicated by a failed batch does not sit unresolved.
- **R10.** A request that fails goes back to its workflow; every default is set from measured rows.

**Ruled by Leo, 2026-11-20, in this session.**

- **L1. Targeted re-run.** On a red batch the failing checks are re-run on each member alone. A member that fails
  is finalized as a verify failure. The rest get a very low failure probability and go back to the pool.
- **L2. One rule.** Batch size is one rule computed from the available information; it supersedes the depth
  table, and the cost of a red batch is a term of it.
- **L3. Any request can be sent back**, hand-submitted included, as it can fail any other way. A scan conviction
  counts for the merge-thrash ladder like any verify failure.
- **L4. Sequencing.** The scan chain of section 8, from ι0 to β2, lands ahead of task 5041's extraction, which
  waits for β2. Task 10995 may wait for this PRD's first task.
- **L5. The 2026-11-18 ruling is amended.** The scan and its fallback replace "every member of a red batch
  verifies alone" (decision D6 of `plans/merge-lane-policy-prd.md`), and the rule replaces its depth table (D7).
- **L6. Measure.** Fewest verifies per landing.
- **L7. Lane and scheduler state survive restarts, as an enforced rule:** on the checked design-invariant list
  now, with a declared register and a guard for the merge lane now, and a design gate for the scheduler (ι1 to ι3).
- **L8. The split.** The statistical design of draft 3 (joint inference, deferral, main as an unknown, conviction
  by elimination with a threshold from a cost model) is behind a design gate (η), with what an entry is built on,
  a narrower cascade and larger caps. Leaving it out of phase 1 costs 7 to 14% in the simulation.
- **L9. The fallback** for a red batch the scan cannot settle: each member verifies alone, and once one of them is
  blocked the others return as new requests.
- **L10. Phase 1 covers the two main reasons a red batch could not be scanned:** test ids the flake gate could
  not map, and a red lint or type-check leg.
- **L11.** Host quarantine and the drift halt are no longer applied (ruled in another session; task 11000).
  "Always stack" is task 11002.
- **L12. The lane state that no task owned is preserved too** (κ): the count of conflict bounces; the retry
  counters (a landing race, a gate re-verify, a timeout, a full disk, a narrowed retry) and the chain-generation
  counts; the window of recent verify outcomes and the probe round counter; and an operator's halt.
- **L13. Housekeeping:** tasks 10624, 8932, 10741, 10742 and 10753 are cancelled; the dependency changes and
  dated notes of section 8 are made.

**Decided here.**

- **D1. Each member is tested alone on the root.** Its result then depends on nothing but itself and main: a
  second culprit is not hidden behind the first, a member is never blamed for a failure an earlier member causes,
  and nothing depends on whether the unlanded work under the batch lands.
- **D2. Three trees are run first: the root, the base when the batch is stacked, and the tip.** A check that
  fails on the root or the base is attributed to nobody. A check that does not fail again on the tip is dropped,
  as the flake gate drops a test that passes when re-run. The tip's run is the confirmation that the failure is
  real on the scan's host; the flake gate's own verdict is not required. If nothing fails on the tip, the batch
  takes the fallback: this is where a failure that only happens on the laptop ends.
- **D3. The scan runs on the workstation, off the verify hosts, one scan at a time**, in the role and isolated
  form of the flake gate's re-run (`verify.py::_node_id_run`), which takes no verify-admission slot. Member trees
  run four at a time. The root, the base, the tip and every confirming second run run with nothing else of the
  scan running. At most four scans wait; a red batch beyond that takes the fallback.
- **D4. A conviction needs the check to fail twice on the member's own tree and to pass, or not exist, on the
  root and on the base.**
- **D5. Cleared is tied to the branch tip.** A request whose tip changes is no longer cleared. No content
  identity is needed.
- **D6. Scanning is a setting per project, off by default.** dark_factory sets `scan: auto`: scanning is on while
  one scanned tree costs at most 0.4 of a verify on the lane's own rows. When it switches itself off the lane
  files one escalation, and it stays off until an operator sets it on again; nothing measures the cost while no
  scan runs. A tree's cost counts everything it takes: the merge, the environment and both runs. For dark_factory,
  at about 0.05, this is a backstop that is not expected to fire.
- **D7. The rule's rates are fitted from the lane's own rows and start from this document's figures.**
  `merge_evidence.fit_rates: false` pins them at their starting values.
- **D8. Caps of 12 new requests and 24 in total.** 12 is the deepest the lane's table has allowed. The rule may
  choose less. Raising them is η's.
- **D9. What still ends in the fallback:** a failure of the unscoped type-check gate and a red test leg with no
  readable ids, which are not scanned; and a file that fails to collect, which is scanned on three trees and
  stays unresolved.
- **D10. A batch whose owner has no module is scanned and never cleared.** Its ids are mapped with the project's
  full module set. Its verify stopped at the first failing module, so a member that passes the scan has not been
  shown to pass the rest: it is released as a new request.
- **D11. A request the scans release twice verifies alone**, whatever the reason: the root or the base failed, its
  own tree could not be judged, or its batch's verify was incomplete.
- **D12. A batch whose verify gives no evidence** (a timeout, an infrastructure failure, a skip) takes the
  fallback. A verify that raises sends its owner to verify alone, as today. A trivial pass changes nothing.
- **D13. In a batch that takes the fallback, the new members verify alone and the cleared members become new
  requests again.** A cleared request that is in fact the culprit is then caught the next time round; kept
  cleared, it could ride and break batch after batch. This holds for every fallback, a verify with no evidence
  included. It narrows L9's "each member verifies alone" to the new members.
- **D14. A batch's first kind is sized directly.** The batch takes as many requests of the front's kind as make
  the cost per member smallest. Requests of the other kind then join one at a time while each adds less than it
  would cost elsewhere. At the starting rates a batch of 12 new requests takes the cleared ones waiting up to a
  defect rate of 0.08 and none from 0.11; a batch started by a cleared request takes 12 new ones up to 0.17 and
  none at 0.20. Together the two kinds cost 9 to 22% less than apart, so a batch started by a new request leaves
  that unused from 0.11. Sizing both kinds together is η's.
- **D15. Reify and every other project keep scanning off** until they opt in. The rule then sizes their batches
  for the release rule: 7, 5, 4 and 3 new requests at defect rates 0.05, 0.08, 0.14 and 0.20, against the table's
  5, 4, 3 and 3.
- **D16. New lane code goes in new modules** inside the merge-lane ratchet's ceilings; each leaf records its raise.

**Considered and rejected** (measured in `evidence/REPORT.md`).

- *The statistical design as phase 1, or as the fallback* (draft 3). With the scan it had nothing to do in the
  simulation except as the fallback, where unchanged it loses 82 to 96% and repriced it gains 7 to 14% over L9.
  It is the larger and riskier build. It is η's.
- *Re-verifying the members of an unscannable red batch in halves.* 2 to 14% better than each alone, 5 to 6%
  behind L9's release rule, and it needs more state.
- *Testing each member on the batch's base* (draft 6). One tree fewer for a stacked batch; but a conviction would
  then rest on unlanded work that may never land.
- *Scanning the trees the batch build already made* (draft 4: the base plus the first n members). No new merges,
  and a binary search is possible; but a second culprit that breaks the same check is hidden behind the first,
  and a later member can be blamed for a failure an earlier one causes. It overtakes member-alone trees only when
  a tree costs 15 to 20% of a verify, which is η's case.
- *Deciding per batch whether to scan, by comparing its trees with what its fallback would cost* (draft 6). The
  fallback's cost depends on the going cost, which depends on the decision: the computation has no fixed point
  for tree costs of 0.3 to 0.8, and near 0.4 it sizes a batch at one member. D6's test has one answer per project.
- *Running the scan on the verify host before its lease is released.* The worktree is removed and the lease
  released before the lane sees a red (`post_merge_verify.py::_PostMergeVerify.run`;
  `worker.py::_vacate_inflight_entry`), and the laptop cannot run a subset. Off the hosts the simulated lane
  lands about 60% more.
- *One isolated failure as proof.* See "flaky tests are common". The second run costs 4 to 6%.
- *Extending the `merge_verify` row.* It is written per attempt, before the verdict is final.

## 4. Pre-conditions (G3)

Verified on main, by symbol:

| Assumed | Evidence |
|---|---|
| A batch's base, its owner's tree, and each link's merge commit | `merge_lane/types.py::RealMergeItem.base_sha`, `.merge_result.merge_commit`, `.merged_branch_tip`; `ChainResult.links`, `link_request_ids`, `tip`; a link is merged `--no-ff` by `worker.py::build_chain`, so its tip is its merge commit's second parent |
| A base can be an in-flight entry's tip or its owner's own commit | `merge_lane/remerge_base.py::entry_bases`; `worker.py::_frontier_base_commits` |
| A worktree at any commit, provisioned as the merge gate provisions its own | `git_ops.py::GitOps.ephemeral_worktree(kind, sha, warm_seed=)`; `verify.py::_provisioned_ephemeral_worktree`, `_preprovision_shared_venv` (its skip is keyed by worktree path) |
| Worktree kinds and their protection from the reaper | `git_ops.py::WorktreeKind`, `PROTECTED_PREFIXES`; `orchestrator/scripts/warm-lane/lib_lane_state.sh` (`LANE_PROTECT_GLOB_FALLBACK`) |
| Run a set of test ids on a tree, by module, serially, with a per-test timeout, statuses read from junit | `verify.py::run_node_ids`, `_run_node_id_group`, `_node_id_run`, `_node_id_test_config` (`NodeIdStatus`: pass, fail, error, skipped) |
| How test ids are mapped to modules | `verify.py::_group_node_ids_by_subproject` (it probes the tree for the file and takes the first module that has it) |
| The project's full module set | `config.py::OrchestratorConfig.module_configs_or_empty`; a batch whose owner has no module gets the empty set (`merge_lane/post_merge_verify.py::merge_boundary_module_configs`) |
| The failing ids of a red verify, as the flake gate holds them and as the test output prints them, from either host | `flake_ledger.py::FlakeSuppression.test_ids`, `unconfirmable_reason`, `flake_suppression_from_wire`; `verify.py::_extract_failing_test_ids`; `VerifyResult.test_output` |
| Which legs were red, and whether the verify was the global command | `verify_evidence.py::VerifyUnit` (`module:<m>:lint`, `module:<m>:type`, `module:__global__:test`, `ran_red`); `verify.py::VerifyResult.test_leg_failed` |
| Four runs at once do not queue behind each other | the merge role takes no admission slot (`shared/src/shared/verify_admission.py::_GATED_ROLES`) |
| The final verify result exists in one place for every path | `post_merge_verify.py::_PostMergeVerify.run`, after the cross-check |
| An additive way to hand a result to the lane | `InflightVerifyRun._launch`'s `on_result`; `shadow_baseline_sink` |
| The lane learns a verdict synchronously, before the links are released | `inflight_verify.py::InflightVerifyRun._take_verdict`, then `run`'s `finally` → `_release_chain_resources` |
| Where a red batch is settled; where a raising batch is guarded; where a single's red is delivered | `InflightVerifyRun._settle_chain`, `_on_verify_error`; the fail arm of `worker.py::SpeculativeMergeWorker._finalize_inflight` |
| The mark and its readers | `SpeculativeMergeWorker._next_dispatch_unchained`; `chain_snapshot`; guard 2b of `_deep_chain_placement`; `_retire_item` |
| A red batch's owner goes back through the arrival queue | `worker.py::_requeue_request` |
| One pick; one predicate hides a request from the pick, the bounce and coalescing; the merger can remove a waiting request | `intake_buffers.py::MergeIntake.pop_next_pickable`, `_clique_minimal_visible` (`is_reserved`), `remove_buffered`; `worker.py::_resolve_or_drop_abandoned` |
| A request's outcome is observable where the journal entry is removed | the result future's done-callbacks in `merge_lane/intake.py` and `intake_buffers.py` |
| Waking the merger | `SpeculativeMergeWorker._resume_signal` (as `ChainReservations(on_release=...)` does) |
| The outcome a red single gets, and the thrash signature | `merge_lane/verify_failure_outcome.py::red_verify_outcome`, `_task_fault_outcome`, `_unscoped_gate_outcome`, `RedVerifyContext`; `shared/task_metadata.py::RetryLedger.compute_merge_outcome_signature` (category and cause hint first) |
| A detached task off the verify path, registered for shutdown | `merge_lane/main_health_probe.py::spawn_main_health_probe` |
| A persistent set with a file behind it | `guard_state.py::PersistentSet` |
| Rows are read back at start; a new event is one enum member | `event_store.py::EventStore.fetch_recent_events_by_type`, `EventStore.emit`; precedent `seed_chain_policy` |
| Where batch size and members are chosen | `_deep_chain_placement` → `merge_lane/chain_policy.py::choose_depth`; `_reserve_and_build_chain` reserves a prefix of the snapshot |
| Deep test scenes pin a size through the table | `orchestrator/tests/_merge_deep_scene.py` |
| The design-invariant list, its lock-step sites and the ratification table | `docs/legibility/design-invariants.md` (INV-14 is the last), `docs/legibility/design-invariants-fixtures.md`, `skills/prd/references/gates.md`; `tests/repo_guards/test_design_invariants_consistency.py::PINNED_SITES`; `docs/legibility/exception-ratifications.yaml` |

Not on main; built here unless a producer is named:

| Needed | Delivered by |
|---|---|
| The tested tree on an in-flight entry: its root, its base, and each member's merge commit and tip | α1 |
| The final verdict's facts reaching the lane for a pass as well as a red (a pass returns `None` today) | α2 |
| A node-id run that takes ids already mapped to modules and returns each module run's result; whether a tree's own collection contains a test | σ1 |
| A worktree kind for scan trees; removal of trees of that kind left by a dead process (today's kinds leak: `_mainprobe-*` directories from July are still present); merging a commit into such a worktree | σ1 |
| One module's lint or type-check leg run alone on a tree | σ3 |
| Reads of the event store that raise on failure; a write that reports failure (`fetch_recent_events_by_type` returns `[]` on error; `emit` returns nothing) | ρ |
| A fast landing walk for large batches | task 10998 (pending, high, no dependencies) |

Used if present, not required: task 9144 (`MergeOutcome.kind`), task 9182 (one terminal hook), task 10995.

## 5. Cross-PRD relationships and seam owners (G4)

| Other work | Direction | Seam | Owner | Action |
|---|---|---|---|---|
| Tasks 10617, 10723, 8528, 10920; `merge-lane-policy-prd.md` D6, D7 | supersedes | what follows a red batch; batch depth | this PRD (L5) | ρ and σ2 add the dated delta note to D6; β2 removes `chain_policy.py` and `depth_table` with one on D7 |
| Task 10624 (mark order, coalescing honours the mark) | absorbs | the same sites | this PRD | item 1 is ρ's "set before the links are released"; item 3 is ρ's coalescing rule; item 2 is moot. Cancel at decompose |
| `plans/deep-merge-ahead-prd.md` decision 9 | amends | a conviction feeds the thrash ladder | σ2 | dated delta note |
| Task 7918 (`run_node_ids`); tasks 8882, 9129, 9132, 9134, 9922, 10240, 4585, 5292, 6290, 6506, 8410, 6869, 8695, 8878, 8912 | consumes; same code | the isolated re-run and its helpers in `verify.py` | each its own | σ1 is the first production caller of the node-id run and uses the helpers as they stand when it is implemented; the file lock serialises. Dated note on 10240 (re-run on the failing leg's host): the scan stays on the workstation |
| Tasks 7919, 7922, 7053 | none | checks run on a tree before or outside a verify | each its own | 7922 searches a batch's own prefix commits before the verify; the scan tests members alone after it |
| Task 7616 (batches whose owner has no module get no junit) | none | their failing set is incomplete | 7616 | until it lands they are scanned and never cleared (D10) |
| Tasks 7921, 9171, 9156; gate 10577 | produces for | "the root, a main commit, fails these checks" | the main-health probe owns "main is red" | σ2 spawns the existing deferred probe when a scan's root fails; a red batch no longer spawns it for its owner. 7921's "held" is a different state from this PRD's "in scan". Dated note on 7921 |
| Task 9908 (`merge_verify.chain_request_ids`) | none | what an attempt dispatched | 9908 | unchanged |
| Task 7824 (build the batch inside the verify task) | none | when a batch's members are known | 7824 | α1 publishes the tree when the build returns, wherever the build runs. Dated note |
| Tasks 10995, 11002, 7810 | produces for | which requests a tree contains, by merge commit | α1 builds `InflightEntry.tree`; they read it | 10995 depends on α1. Dated notes on 10995 (its containment walk reads the tree) and 11002 (the tree carries merge commits and tips, no content id) |
| Task 10998 (skip the post-landing type-check) | consumes | landing 12 members takes 45 minutes today | 10998 | β2 depends on 10998 |
| Task 10721 (a passed head re-verified alone) | consumes | a re-verify of a rebased head takes no members | 10721 | the rule adds nothing to such a dispatch |
| Tasks 8933, 8329, 10136, 9133, 8359, 9268, 9269 | same files | `worker.py`, `verify_dispatch.py`, `types.py`, `git_ops.py` | each its own | serialised by the file lock |
| Task 10881 (share the tree-recording test pair) | overlaps | `orchestrator/tests/_merge_lane_fakes.py` | 10881 | α1 moves the tree-recording pair there if 10881 has not; dated note on 10881 |
| Task 5041 (extract `speculation.py`) | must follow | the placement code | 5041 moves; this PRD edits first (L4) | 5041 depends on β2; gate 10573 re-points names |
| Part 1, task 9935 (pool order) | amends | "task 8528's suspect list" | 9935 owns `pool_order()` | 9935's text: no fronted list exists; a request in scan, verifying alone or cleared is ordered like any request; the pick filter keeps ρ's predicate. 9935 depends on ρ |
| Task 9182 (one terminal hook) | consumes | a request leaving the lane | 9182 | ρ calls `forget` in `_retire_item`; 9182 carries that onto its hook. Dated note |
| Tasks 9144, 9207 | none | the outcome kind and block table | each its own | a scan conviction is the existing post-merge-verify block; no new kind |
| Task 9766 (moves `EventType`) | none | where event names live | 9766 | new event types are declared in `EventType` wherever it lives at dispatch |
| Tasks 5411, 10996 (docs gate, docs lane) | consumes | a docs-lane verify is not a full gate | 10996 | such a request is never a batch member; dated note |
| `plans/merge-chain-visibility-prd.md`; tasks 10759, 10761, 10762, 10621 | amends | "isolation", `unchained_mark`, the `chain_policy` snapshot block and the depth-basis vocabulary are defined by the mark and the table | this PRD defines what follows a red batch and how a batch is sized; that PRD records and shows it | θ: a design gate that revises that PRD. 10759, 10761, 10762 depend on θ |
| Tasks 10741, 10742, 10753 (small fixes to the chain-policy code) | superseded | code β2 deletes | — | cancel at decompose |
| Tasks 9176, 4838, 5623; gate 10217 leaf C | none; same file for κ | lane holds and the journal across a restart | each its own | named as owners in ι2's register. κ keeps an operator's halt across a restart on today's `HaltState`, with the operator as its owner; dated notes on 9176 (carry it into the holder model) and 4838 (a restored operator halt is not an ownerless one) |
| Task 8532 (7-day readout) | none | reads the lane as deployed | 8532 | dated note: ι0 and ρ may deploy inside its window |
| Task 8932 (idle-host hedge, dense table) | superseded | the depth table; what a spare host verifies | this PRD; η | cancel at decompose |
| Gate 10572 rulings | consumes | merge verifies take spare capacity | 10572 | the lane stays work-conserving |

## 6. Contract (B + H)

### 6.1 The tested tree

```
@dataclass(frozen=True)
class TreeMember:
    request_id: str
    task_id: str
    merge_commit: str        # the tree with this member and everything before it
    tip: str | None          # branch tip merged into that tree; None for an owner that is a rebased head
    own: bool                # True: a member of the verifying entry; False: of an entry it was built on

@dataclass(frozen=True)
class TestedTree:
    base_commit: str                         # the commit the owner was built on
    root_commit: str | None                  # the main commit under the batch; None if it cannot be resolved
    members: tuple[TreeMember, ...]          # built-on members first, then own members in land order
```

- `InflightEntry.tree` is published once, through the entry slot (`InflightEntrySlot`), when the entry's build
  returns and before its verify starts. It is never mutated.
- `base_commit` is the owner item's `base_sha`. It is resolved by commit, never by position in the frontier:

  | `base_commit` is | Built-on members | `root_commit` |
  |---|---|---|
  | a commit main contains | none | `base_commit` |
  | the `merge_commit` of a member of a live in-flight entry's tree (its tip, or its owner's own commit) | that entry's members up to and including that member | that entry's `root_commit` |
  | anything else (task 8933's class) | none | `None` |

- Verifies with no in-flight entry are outside this contract: a train's inline verify and its gate re-verify,
  `worker.py::reverify_member_solo`, `VerifyDispatcher.gate_history_leg`, shadow verifies.
- **T1.** `members` lists each request at most once; the first own member is the owner.
- **T2.** The last own member's `merge_commit` is the tree the verify ran on.

### 6.2 The verdict's facts

```
class Evidence(StrEnum): PASS, RED, NONE
class NoEvidence(StrEnum): TIMED_OUT, INFRA, TRIVIAL, SKIPPED
class CheckKind(StrEnum): TEST, LINT, TYPE
class ScanBlocker(StrEnum): UNSCOPED_GATE, NO_CHECKS, NO_IDS, LEG_RED

@dataclass(frozen=True)
class Check:
    kind: CheckKind
    module: str | None       # the module of a leg; None for a test id, which the scan maps
    test_id: str | None      # set exactly for a TEST

@dataclass(frozen=True)
class FailingChecks:
    checks: tuple[Check, ...]
    complete: bool                   # every module's test leg ran to a verdict
    blocker: ScanBlocker | None      # why this red cannot be scanned; None when it can

@dataclass(frozen=True)
class VerdictFacts:
    evidence: Evidence
    no_evidence: NoEvidence | None   # set exactly when evidence is NONE
    failing: FailingChecks | None    # set exactly when evidence is RED
    duration_secs: float
    host: str

def verdict_facts(result: VerifyResult, *, duration_secs: float, host: str) -> VerdictFacts
```

- `verdict_facts` is total and pure, over the final result of a verify:

  | Final verify result | Evidence |
  |---|---|
  | passed, suite ran | `PASS` |
  | passed, `trivial` | `NONE`, `TRIVIAL` |
  | timed out, the unscoped type-check timeout included | `NONE`, `TIMED_OUT` |
  | red in an infra class: ENOSPC, transient infra, lock contention | `NONE`, `INFRA` |
  | red, anything else | `RED` |

  The infra test is one predicate shared with `verify_failure_outcome.py::red_verify_outcome`.
- The `TEST` checks are the flake observation's ids when it holds any; otherwise they are read from the result's
  test output with the flake gate's own reader (`verify.py::_extract_failing_test_ids`). The `LINT` and `TYPE`
  checks are the units that ran red.
- `blocker`, first match: the unscoped type-check gate is what failed, `UNSCOPED_GATE`; there is no check at all,
  `NO_CHECKS`; a test leg is red and there is no `TEST` check, `NO_IDS`; there is a `LINT` or `TYPE` check,
  `LEG_RED`. σ3 removes the last.
- `complete` is false when the verify ran the global test command (a `module:__global__:test` unit), when the
  flake observation reports a truncated session or a leg without a verdict, or when a module's test leg wrote no
  report.
- The facts and the result they were read from reach the lane through a sink set per dispatch, fired once in
  `_PostMergeVerify.run` after the cross-check, for a pass as for a red. `run_post_merge_verify`'s return type
  does not change. The facts stay on the in-flight entry until it is finalized.
- The lane writes one `merge_observation` row per verdict of a dispatched entry, in
  `InflightVerifyRun._take_verdict`: an observation id, the entry's `TestedTree`, the facts. The low-disk skip,
  which never dispatches, is written as `SKIPPED`. A verify that raises or is cancelled writes none.
- Wire types and decode live in `shared/src/shared/merge_evidence.py`; an unknown enum value raises with field and
  value, and a reader counts and skips such a row. `verdict_facts` lives in `merge_lane/observation.py`.
- **V1.** A `NONE` verdict clears nothing and convicts nothing. Under D12 it can make a cleared member new again.

### 6.3 The scan

A module with no lane state, `merge_lane/culprit_scan.py`:

```
class CheckStatus(StrEnum): PASS, FAIL, ABSENT, ERROR
class TreeProblem(StrEnum): NO_TREE, MERGE_FAILED, PROVISION_FAILED, OVER_BUDGET

@dataclass(frozen=True)
class ScanMember:
    request_id: str
    tip: str | None
    own_tree: str | None            # a commit that is the root plus this member alone, when one exists

@dataclass(frozen=True)
class ScanInput:
    observation_id: str
    root_commit: str
    base_commit: str                # equal to root_commit for a batch built on main
    tip_commit: str                 # the tree the verify ran on
    members: tuple[ScanMember, ...] # own members in land order
    checks: tuple[Check, ...]
    complete: bool

@dataclass(frozen=True)
class FailureEvidence:
    category: str
    cause_hint: str | None
    report: str

@dataclass(frozen=True)
class TreeRun:
    tree: str                                   # 'root', 'base', 'tip' or a request id
    statuses: Mapping[Check, CheckStatus]       # first run
    confirm: Mapping[Check, CheckStatus]        # second run of the checks that failed in the first
    failure: FailureEvidence | None             # of the first failing run
    secs: float
    problem: TreeProblem | None

class ScanOutcome(StrEnum): ATTRIBUTED, CONTROL_FAILED, NOT_REPRODUCED, INCOMPLETE
class MemberVerdict(StrEnum): CULPRIT, CLEARED, RELEASED, FALLBACK

@dataclass(frozen=True)
class ScanVerdict:
    outcome: ScanOutcome
    members: Mapping[str, MemberVerdict]
    culprit_checks: Mapping[str, tuple[Check, ...]]
    main_failed: tuple[Check, ...]              # fail on the root
    ahead_failed: tuple[Check, ...]             # pass on the root, fail on the base
    dropped: tuple[Check, ...]                  # failed in the verify, did not fail again on the tip

async def run_scan(git_ops, config, module_configs, scan: ScanInput, *,
                   tree_budget_secs: float, concurrency: int) -> tuple[TreeRun, ...]
def attribute(scan: ScanInput, runs: Sequence[TreeRun]) -> ScanVerdict
```

**Running.**

- `module_configs` is the project's full module set. Each test id is mapped to a module once, on the tip tree,
  with the mapping the flake gate uses; every tree runs it in that module. An id whose path exists in more than
  one module is mapped to the one whose test unit ran red in the verify. An id that maps to no module, or to more
  than one, is `ERROR` on the tip and is run nowhere.
- Each tree gets a fresh worktree of kind `WorktreeKind.CULPRIT_SCAN`. The owner's own tree is its merge commit
  when the batch was built on main. Any other member's own tree is made by merging its `tip` onto the root in
  that worktree: a merge that does not complete is `MERGE_FAILED`, and a member with neither is `NO_TREE`. The
  worktree is then provisioned as the merge gate provisions its own (`_preprovision_shared_venv` with the merge
  role's environment and timeout).
- On each tree, a test id is `ABSENT` when its file is not in the tree, or when the file collects there and does
  not contain the test. The other ids run by module, serially, in the form of `verify.py::_node_id_test_config`
  under the merge role: `pass` is `PASS`, `fail` is `FAIL`, `error` and `skipped` are `ERROR`, as is every id of a
  file that does not collect. From σ3 a leg is run as the module's lint or type-check command alone: `PASS` or
  `FAIL` by its exit, `ABSENT` when the module is not in the tree, `ERROR` when it could not run. Checks that
  `FAIL` are run a second time on the same tree.
- Order: the root, the base when it differs from the root, and the tip, one after the other. The *live* checks
  are those that `FAIL` on the tip and are `PASS` or `ABSENT` on the root and on the base. Then the members,
  `concurrency` at a time, on the live checks only. A second run starts when no other tree of the scan is
  running. With no live check, no member tree is run.
- `tree_budget_secs` bounds one tree's runs, the second included; a tree that exceeds it is `OVER_BUDGET`.
  Worktrees are removed on every exit. `GitOps.reclaim_leaked(kind)` removes worktrees of a kind that no live
  process owns; the lane calls it before its first scan after a start.

**Attributing** (pure). On the tip a check that is `PASS` is *dropped* and one that is `ERROR` is *unresolved*. A
check that fails on the tip is *main's* when it is `FAIL` on the root, *ahead's* when it is not main's and is
`FAIL` on the base, *unresolved* when it is `ERROR` on either, and otherwise *live*. A member's own tree *fails*
a live check when both its runs are `FAIL`, *passes* it when the first run is `PASS` or `ABSENT`, and is *unsure*
otherwise; a tree with a `problem` is unsure about every check. A live check is *explained* when some member's
tree fails it.

| Case | Outcome | Each member |
|---|---|---|
| The root, the base or the tip has a `problem`, or the scan did not finish | `INCOMPLETE` | `FALLBACK` |
| No check fails on the tip | `NOT_REPRODUCED` | `FALLBACK` |
| Every check that fails on the tip is main's or ahead's | `CONTROL_FAILED` | `RELEASED` |
| Otherwise | `ATTRIBUTED` | by the rules below, first match |

1. `CULPRIT`: its tree fails a live check.
2. `FALLBACK`: some live check is not explained, or some check is unresolved.
3. `CLEARED`: its tree passes every live check, no check is main's or ahead's, and `complete` is true.
4. `RELEASED`: otherwise.

- **N1.** No member is `CULPRIT` for a check that is not `PASS` or `ABSENT` on the root and on the base.
- **N2.** A `CULPRIT` has two `FAIL` runs of one check on its own tree.
- **N3.** `run_scan` leaves no worktree behind, on any exit.
- **N4.** With checks whose result depends on one member each, no control failing and no `problem`, every
  defective member is `CULPRIT` and every other member is `CLEARED` (`RELEASED` when `complete` is false).

### 6.4 What the lane does with a red batch

State, in `merge_lane/batch_state.py::BatchState` (one owner; `SpeculativeMergeWorker` holds one):

```
class Kind(StrEnum): NEW, CLEARED
class AloneReason(StrEnum): NO_EVIDENCE, VERIFY_RAISED, UNSCANNABLE, SCAN_BUSY, SCAN_FALLBACK, RELEASED_TWICE

class BatchState:
    def begin_scan(self, scan: ScanInput) -> None
    def apply_scan(self, observation_id: str, verdict: ScanVerdict,
                   evidence: Mapping[str, FailureEvidence]) -> None
    def abandon_scan(self, observation_id: str) -> None
    def verify_alone(self, request_ids: Collection[str], *, red_set: str | None, reason: AloneReason,
                     scanning: bool) -> None
    def unclear(self, request_ids: Collection[str], *, red_set: str | None, scanning: bool) -> None
                                                                  # cleared requests that are new again
    def note_blocked(self, request_id: str) -> tuple[str, ...]    # the others of its red set, now released
    def note_landed(self, request_id: str) -> None
    def clear(self, request_id: str, tip: str) -> None
    def conviction_applied(self, request_id: str) -> None
    def forget(self, request_id: str) -> None                     # the request left the lane
    def in_scan(self, request_id: str) -> bool
    def alone(self, request_id: str) -> bool
    def kind(self, request_id: str, tip: str) -> Kind
    def pending_convictions(self) -> Mapping[str, Conviction]     # request id -> (tip, evidence, checks)
    def open_scans(self) -> tuple[ScanInput, ...]
```

- **Alone** is what the un-chained mark means today: the request takes no members and is nobody's member. It
  outranks cleared. It ends when the request leaves the lane.
- **The release rule (L9).** Requests sent alone together carry their `red_set`. When one of them is resolved
  `blocked` on its own red verify, the lane calls `note_blocked` and the others of the set stop being alone: they
  are new requests again, not cleared. A request sent alone with no red set stays alone until it leaves. The
  trigger is in the fail arm of `_finalize_inflight`: the entry's own facts are `RED` and its outcome is
  delivered.
- **The fallback** for a set of members (D13): those that are new verify alone,
  `verify_alone(new_members, red_set=<observation id>, reason=..., scanning=...)`, and those that are cleared
  become new, `unclear(cleared_members, red_set=<observation id>, scanning=...)`. Both rows carry the observation
  and whether scanning was on.
- **In `_take_verdict`, before the links are released**, for an entry with two or more own members:

  | The verdict's facts | What happens to the own members |
  |---|---|
  | `PASS` | nothing here; those that then do not land are cleared at their tips |
  | `NONE`, `TRIVIAL` | nothing |
  | `NONE`, otherwise | the fallback, reason `NO_EVIDENCE` (D12) |
  | `RED`, and scanning is off, or there is a `blocker`, or `root_commit` is `None` | the fallback, reason `UNSCANNABLE` |
  | `RED`, and `scan_queue_max` scans are already waiting | the fallback, reason `SCAN_BUSY` |
  | `RED`, otherwise | `begin_scan`; the scan is queued |

  A verify that raises sends its owner alone with no red set, as `_on_verify_error` marks it today. The owner is
  requeued as today. Before σ2 lands, every `RED` row is the fallback. Before β1 lands, `scan: auto` acts as `on`.
- **A request in scan** is hidden from the pick, the suffix bounce, batch building and coalescing. Three sites
  read it: `MergeIntake`'s `is_reserved`, and the two direct reads of `ChainReservations.holds` in
  `chain_snapshot` and in coalescing. A request with a pending conviction stays in scan until the conviction is
  applied or lapses. A request that verifies alone is not a candidate member and is not coalesced.
- **Running scans.** The lane runs scans through a port, `ScanRunner`, whose production form is `run_scan` and
  whose test form answers by tree. One scan runs at a time, detached from the verify path and registered for
  shutdown as the main-health probe is. A supervisor abandons a scan whose task has died, or that has run longer
  than `tree_budget_secs` times twice its number of trees: every member takes the fallback, reason
  `SCAN_FALLBACK`. When a scan is applied or abandoned the merger is woken.
- **Applying a scan's verdict** is one call, `apply_scan`, with no await inside it:
  - `CULPRIT`: a pending conviction for that request and tip, with its own tree's `FailureEvidence`.
  - `CLEARED`: cleared at its tip.
  - `RELEASED`: the request's count of releases rises, and on the second it is alone, reason `RELEASED_TWICE`
    (D11). A cleared request that is released is new again.
  - `FALLBACK`: the members with that verdict take the fallback, reason `SCAN_FALLBACK`.
- **A pending conviction is applied by the merger before a pick**, at both of its pick sites
  (`_acquire_next_request` and `_look_ahead`). When the request is waiting in a buffer with
  that tip, it is removed (`remove_buffered`) and resolved with the outcome a red single gets
  (`_task_fault_outcome`'s shape): `blocked`, `block_category=POST_MERGE_VERIFY`, the failure category, cause hint
  and report of its own tree's failing run, and one sentence saying how it was found. Its dry run is spawned as
  for a red single. When the request shows another tip, or leaves the lane, the conviction lapses and the request
  is no longer in scan.
- **When the root fails** (`main_failed` is not empty), the existing deferred main-health probe is spawned for
  the owner, with the verify's result kept beside the scan. After a restart that result is gone and no probe is
  spawned.
- **A red batch's owner gets no dry run and no main-health probe of its own.** `RedVerifyContext` gains the
  entry's number of own members, and no arm of `red_verify_outcome` spawns either when it is two or more.
- **A red single** is handled as today.
- **Outcomes are recorded** with the kind the request had when the outcome was delivered. `note_blocked` and
  `conviction_applied` are called as above. `note_landed` is called from the request's done-callback when the
  outcome's status is `done`; by then `forget` has run, so `forget` keeps the request's kind until that callback
  has read it. The callback drops the kept kind whatever the status, and one it never reads is dropped at the next
  checkpoint. A request that a scan clears, or that is cleared because its passed batch did not land, is
  recorded as a sound outcome of a new request at that moment.
- **A `merge_scan` row** is written when a scan is applied or abandoned: the observation, each tree's statuses,
  seconds and problem, the verdict, and the time it waited in the queue. A tree's seconds run from the start of
  its merge to the end of its last run. Twenty scans in a row that neither convict nor clear any member file one
  L1.
- **S1.** No request is resolved by a scan while it is merging, reserved or being verified.
- **S2.** A conviction is applied only to the request and tip that were scanned.
- **S3.** A scan always ends: it is applied, abandoned by the supervisor, or run again after a restart (6.6).

### 6.5 The batch rule

Pure modules, `merge_lane/batch_rule.py` and `merge_lane/rule_rates.py`:

```
@dataclass(frozen=True)
class Rates:
    defect_rate: float            # q: a new request is defective
    cleared_defect_rate: float    # r: a cleared request is
    unscannable_share: float      # u: a red batch ends in the fallback although scanning is on
    scan_tree_cost: float         # one scanned tree, as a share of a verify; read by `scan: auto` only
    n: Mapping[str, int]

@dataclass(frozen=True)
class GoingCosts:
    new: float
    cleared: float

def scanning(mode: ScanMode, rates: Rates, *, max_tree_cost: float) -> bool
def fallback_cost(n: int, p: float, costs: GoingCosts) -> float
def batch_cost(new: int, cleared: int, rates: Rates, costs: GoingCosts, *, scanning: bool) -> float
def going_costs(rates: Rates, *, caps: Caps, scanning: bool) -> GoingCosts
def plan_batch(front: Kind, waiting: Sequence[Kind], rates: Rates, *, caps: Caps, scanning: bool) -> tuple[int, int]
def fit_rates(rows: RateRows, *, start: Rates, prior_weight: int) -> Rates
```

All costs are in verifies. A batch has `f` new and `c` cleared members, `n = f + c`; one member costs 1.

- **Scanning** is true under `scan: on`, false under `off`, and under `auto` true while
  `rates.scan_tree_cost <= max_tree_cost` (D6). When `auto` turns false the lane files one L1, deduplicated while
  open.
- **What a red batch is expected to hold.** `P = 1 − (1 − q)^f (1 − r)^c` is the chance it is red,
  `k = (f·q + c·r) / P` the culprits expected when it is, and `p = (f·q + c·r) / n` its mean defect probability.
- **The fallback's cost.** Members verify alone in order until one is blocked, then the rest return as new:
  `F(n) = B + 1 + (n − 1 − B) · A_new` for `n ≥ 1`, which is 1 for one member, and `F(0) = 0`, where
  `B = Σ_{j=1..n} (j − 1)(1 − p)^(j−1) p / (1 − (1 − p)^n)` is the sound members expected ahead of the first
  culprit; with `p` of 0, `F(n) = n`.
- **The batch's cost.** `C(f, c) = 1 + P · [ s · (n − k) · A_cleared + (1 − s) · (F(f) + c · A_new) ]`, with
  `s = 1 − u` when scanning and 0 when not, and `C = 1` when `P` is 0. After a scan the culprits are gone and each
  sound member still owes `A_cleared`. In the fallback the new members verify alone, `F(f)` at `p = q`, and each
  cleared member is new again and owes `A_new` (D13). The scan's own time is off the verify hosts and is not a
  term.
- **Going costs.** `A_cleared` is the smallest `C(0, c) / c` over `c = 1 .. max_total`, and `A_new` the smallest
  `C(f, 0) / f` over `f = 1 .. max_fresh`. Each contains the other. They are solved together by iteration from 1
  and 1, to a change under 1e-9; if 500 iterations do not reach it, the rule returns a batch of the front alone
  and the lane counts it on the observation.
- **Members (D14).** The dispatching request is the front. The batch takes as many waiting requests of the
  front's kind, in pool order, as make `C / n` smallest, the larger number on a tie, within the caps. Waiting
  requests of the other kind then join in pool order, one at a time, while the cost each adds is below the going
  cost of its kind. At most `max_fresh` new members and `max_total` in all. The chosen members keep pool order
  and need not be adjacent in it; the reservation and `build_chain` are given exactly them.
- **Not candidates:** a request in scan, one that verifies alone, a docs-lane request, a train, a request still
  stacked on an unlanded base. A request that verifies alone and a re-verify of a rebased head (task 10721) take
  no members.
- **Rates.** `fit_rates` is pure, over the newest 600 outcome rows of `BatchState`, 150 `merge_scan` rows and 300
  `merge_observation` rows, which the lane reads at start and keeps current:

  | Rate | Events over trials | Starts at |
  |---|---|---|
  | `defect_rate` | new requests blocked on their own red verify or by a conviction, over those, new requests that landed, and new requests recorded sound when they were cleared | 0.14, measured |
  | `cleared_defect_rate` | cleared requests blocked on their own red verify or by a conviction, over those and cleared requests that landed | 0.01, provisional: nothing measures it yet; one interaction in 58 red batches bounds one source below 0.005 |
  | `unscannable_share` | of the red batches of two or more own members that went red while scanning was on: those with any member sent to the fallback, whatever the reason, over all of them | 0.15: 11 of 104 red batches (7 with a red leg, 3 with a file that did not collect, 1 with nested ids) at today's batch sizes, carried to a batch of 12 |
  | `scan_tree_cost` | mean seconds of a scanned tree over mean seconds of a passed verify; the starting value until 20 trees are recorded | 0.05, estimated |

  The first three are `(events + prior_weight · start) / (trials + prior_weight)`. With `fit_rates: false` every
  rate stays at its start, which is a config leaf.
- **Recorded.** `merge_observation` carries the plan made at dispatch: which members were new and which cleared,
  the predicted `1 − P`, the cost, the rates used and whether scanning was on.
  `merge_verify.chain_depth_reason` keeps its keys: `basis` is `cost_rule`, `pass_estimate` is `1 − defect_rate`,
  `row` is null.
- **B1.** With scanning off and `F(n) = n` in place of the release rule's cost, the rule returns the depth of the
  table task 10920 derived, at every pass estimate at least one grid step inside a table row: the table is the
  rule when every member of a red batch verifies alone.
- **B2.** With both defect rates 0 the batch is as large as the caps and the waiting requests allow. With
  `fit_rates: false` that holds across red batches, which is how a test pins a batch size.
- **B3.** `plan_batch` returns what the reference computation returns (`evidence/scripts/rulecheck/rule9.py.txt`).
  At the starting rates with scanning on: a new front with 20 new and 10 cleared requests waiting gives 12 new
  and no cleared member at a defect rate of 0.14, and 12 new and 10 cleared at 0.08; a cleared front with 10
  cleared and 20 new waiting gives 11 cleared and 12 new members at 0.14, and 11 cleared and no new member at
  0.20. With scanning off a new front gives 7, 5, 4 and 3 new members at 0.05, 0.08, 0.14 and 0.20, and a cleared
  front with the same waiting requests gives 11 cleared and 2 new at 0.05 and 11 cleared and none at 0.08.
- **B4.** A free host and a request that can be dispatched always produce a dispatch.
- **B5.** The functions are pure.

### 6.6 Restart

- `BatchState` writes each change as a `merge_batch_state` row in the same step as it applies it. The kinds:
  `scan_begun` (with the whole `ScanInput`), `scan_applied` (with the verdict and each culprit's evidence),
  `scan_abandoned`, `alone` and `uncleared` (each with the observation and whether scanning was on), `released`,
  `cleared`, `landed`, `blocked`, `convicted`, `forgotten`.
- Every `checkpoint_every` changes it writes a `merge_batch_state_checkpoint` row: the cleared tips, the alone
  requests with their red sets, the release counts, the pending convictions and the open scans.
- `EventStore.fetch_events_after(event_types, after_id, limit)` returns rows after a row id, oldest first;
  `EventStore.fetch_newest(event_type)` returns the newest row of a type or `None`. Both raise on failure.
  `EventStore.emit_checked(...)` is `emit` returning whether the row was written.
- `SpeculativeMergeWorker.run` rebuilds before the first pick: the newest checkpoint, the rows after it, and the
  recovered requests. State for a request that was not recovered is dropped. A tail that fills `limit` fails the
  rebuild.
- A scan with a `scan_begun` row and neither `scan_applied` nor `scan_abandoned` is run again; its members are in
  scan again.
- Every start writes a `merge_batch_state_rebuild` row: `ok`, checkpoint age, rows applied, rows skipped.
- If a read raises, the checkpoint cannot be decoded, or more than 5 rows are skipped: the state starts empty,
  the row says `ok: false` with the reason, one L1 is filed, and a fresh checkpoint is written at once. That is
  today's behaviour after every restart.
- Three consecutive failed `emit_checked` writes of `BatchState` file one L1.
- **R1.** The same checkpoint, tail and recovered requests give the same state.

### 6.7 Config, events, snapshot

- `merge_evidence` (own file `orchestrator/merge_evidence_config.py`, mounted once, all leaves hot, each with a
  range validator): `scan` (`off` | `on` | `auto`; default `off`), `scan_max_tree_cost` (0.4), `scan_concurrency`
  (4), `scan_queue_max` (4), `scan_tree_budget_secs` (600), `max_fresh` (12), `max_total` (24),
  `checkpoint_every` (50), `fit_rates` (true), `rate_prior_weight` (20), `start_defect_rate` (0.14),
  `start_cleared_defect_rate` (0.01), `start_unscannable_share` (0.15), `start_scan_tree_cost` (0.05).
- `dark-factory-orchestrator.yaml` sets `scan: auto` (σ2). `merge_deep.chain_cap: 0` stays the switch that turns
  batching off.
- `scan_max_tree_cost` 0.4 is just under the simulated break-even of 0.43 to 0.57. With a tree's runs bounded at
  600 s and a verify of 26 to 42 minutes, dark_factory can only reach it through slow merges or environments.
- `scan_concurrency` 4: in the simulation four trees side by side hold a batch's members 0.25 hours at the median
  against 1.4 for one, and eight add nothing at this tree cost.
- `scan_queue_max` 4: at dark_factory's rates and four trees side by side, a scan waited 0.13 hours at the 95th
  percentile, less than one scan.
- `scan_tree_budget_secs` is two of the per-test timeouts the node-id run injects
  (`verify.py::_NODE_ID_RUN_TIMEOUT_SECS`, 300 s); the slowest type-check leg measured is 220 s. Five tests in the
  repo carry a longer timeout of their own (1,200 or 1,500 s); a tree that needs one of them goes over budget.
- `rate_prior_weight` 20: a rate moves halfway from its start to the rows after 20 trials.
- New event types, declared in `EventType` wherever it lives at dispatch: `merge_observation` (α2);
  `merge_batch_state`, `merge_batch_state_checkpoint`, `merge_batch_state_rebuild` (ρ); `merge_scan` (σ2).
- `SpeculativeMergeWorker.snapshot()`: each in-flight entry gains `tree` (α1); each waiting entry gains `cleared`,
  `alone` and `in_scan` (ρ); the top level gains `evidence: {scanning, scan_queue, rebuilt_ok}` (ρ, σ2) and
  `rates` (β1).

## 7. Boundary-test sketch (B + H)

Driven through the lane's public seam with tree-keyed verdicts and a `ScanRunner` that answers by tree; the scan
engine itself on real git fixtures.

| # | Scenario | Preconditions | Asserts |
|---|---|---|---|
| 1 | The tree is recorded | a batch of 3 is built on an in-flight entry of 2; both return | the batch's `merge_observation` row lists 5 members, 3 of them `own`, each with merge commit and tip, the base and the root |
| 2 | Base cases | an item built on a landed commit that is not main's tip; one built on an in-flight owner's own commit, before that owner took links; one built on a commit no live entry owns | the first has no built-on members and root equal to base; the second's built-on members are that owner alone; the third has no root |
| 3 | No evidence | a batch's verify times out; one is skipped for low disk; one fails for lock contention; one passes trivially | each row has `evidence: none` with its reason; nothing is cleared or convicted; every new member of the first three verifies alone, as one red set each; the fourth changes nothing |
| 4 | The verdict that counts | a verify's remote attempt passes and its cross-check fails | one observation, `RED` |
| 5 | One culprit | a batch of 4 on main is red on test T; T passes on the root and fails twice on the third member's own tree only | the third member is `blocked` with that run's failure report and no verify of its own; the other three are cleared; a `merge_scan` row lists six trees with seconds |
| 6 | Two culprits, one test | as 5, and T also fails twice on the fourth member's own tree | both are blocked |
| 7 | A flaky failure beside a culprit | T fails twice on the third member's tree; it fails once and then passes on the second's | the third is blocked; the second is released as new, not cleared |
| 8 | A flaky failure and no culprit | T fails on the tip, fails once and then passes on one member's tree, and passes on the others | every member verifies alone, as one red set |
| 9 | A test the culprit brought | T's file is not on the root; in another batch T is added to a file that exists on the root; each fails twice on the member that adds it | `ABSENT` on the root and on the other members; that member is the culprit and the others are cleared |
| 10 | A test that cannot run on the root | T reports `error` on the root; T fails twice on one member's tree | nobody is blocked; every member verifies alone, as one red set |
| 11 | The work ahead fails | a batch is built on an unlanded entry; T passes on the root and fails on the base | no member is blocked or cleared; all are released; no main-health probe; no member tree is run |
| 12 | Released twice | the same request is released by a second scan: once because the base failed, once because its own tree would not merge | it verifies alone |
| 13 | Main is red | T fails on the root | no member is blocked; the deferred main-health probe runs for the owner |
| 14 | Part of the failure is the base's | T fails on the base; U passes on the root and the base and fails twice on one member | that member is blocked for U; the others are released, not cleared |
| 15 | A stacked batch | a batch built on unlanded work; T passes on root and base and fails twice on one member's own tree; the work ahead then fails its own verify | the member is blocked; its tree was the root plus itself |
| 16 | Nothing reproduces | a batch verified on the laptop is red on T; T passes on the tip on the workstation | the root and the tip are run; every member verifies alone, as one red set |
| 17 | Fails only together | T fails on the tip, passes on the root and on every member alone | every member verifies alone, as one red set |
| 18 | An owner with no module | a batch whose owner has no module is red in the global test command; its ids map with the full module set; one member's tree fails twice | that member is blocked; the others are released as new, not cleared |
| 19 | An id nobody can map; an ambiguous path | one failing id maps to no module on the tip; another id's path exists in two modules, one of which ran red | the second is run in the module that ran red; the first is unresolved: a culprit found for the second is blocked and the others verify alone, as one red set |
| 20 | Cannot be scanned | a batch fails the unscoped type-check gate; another has a red test leg and no readable ids; another has no root | every member verifies alone, as one red set; no scan runs |
| 21 | A red leg (σ3) | a batch is red on one module's type-check leg, which passes on the root and fails twice on one member's own tree | that member is blocked; the others are cleared. Before σ3 the batch takes the fallback |
| 22 | The release rule | four members verify alone as one red set; the second is blocked on its own red verify | the first has landed; the third and fourth are batched again as new requests, not cleared |
| 23 | Not a release | a member of a red set verifying alone times out; another is cancelled | the others still verify alone |
| 24 | A member that will not merge alone | one link's tip conflicts with the root; a culprit is found among the others | that member is released as new; the culprit is blocked |
| 25 | Side by side | a batch of 8 with `scan_concurrency` 4 | at most four member trees run at once; the root and the tip run before any; each second run runs alone |
| 26 | In scan | a batch is red; its scan waits behind another | its members are not picked, bounced or coalesced and are in no batch until it ends; when it ends the merger picks without a new arrival |
| 27 | The queue is full | `scan_queue_max` scans wait and another batch goes red | its members verify alone, as one red set; no scan is queued for it |
| 28 | A scan that dies or hangs | the scan task is cancelled; a tree hangs | each member verifies alone; no worktree is left |
| 29 | The culprit is the owner | the owner's own tree fails T twice | the owner, requeued, is blocked once it is waiting in a buffer |
| 30 | The request changed | a culprit's branch tip changes before the merger applies the conviction | it is not blocked, and it is picked again as a new request |
| 31 | Picked before the scan | a link whose reservation expired was picked before the verdict and is named culprit | the scan does not resolve it; its own verify decides |
| 32 | Scanning switches itself off | `scan: auto`; 20 scanned trees each cost more than `scan_max_tree_cost` of a verify | the next red batch takes the fallback, one L1 is filed, and the next batch is sized as B3 gives with scanning off |
| 33 | The rule's choices | the cases of B3 | the members B3 lists |
| 34 | Cleared lapses | a cleared request's branch gains a commit | it is counted as new |
| 35 | A pass that did not land | a batch of 5 passes; the landing walk stops after 2 | the other 3 are cleared |
| 36 | The table is a special case | scanning off, each-alone fallback cost, pass estimates inside each table row | `plan_batch` gives the depth task 10920's table gives |
| 37 | Pinning a size | both starting defect rates 0, `fit_rates: false`, `max_fresh` and `max_total` 4, six requests waiting | the batch has 4 members, before and after a red batch |
| 38 | Members need not be adjacent | the pool holds new, cleared, new, cleared and the rule takes only the new ones | the batch is built from the two new requests; the cleared ones keep their places |
| 39 | A raising batch | a batch's verify raises | the owner's next dispatch has no members |
| 40 | Restart | cleared requests, requests alone in a red set, a pending conviction and a red batch whose scan is running are recorded; the orchestrator restarts with the same requests waiting | the snapshot shows the same cleared and alone requests; the conviction is applied; the scan runs again and is applied once; a `merge_batch_state_rebuild` row with `ok: true` |
| 41 | Rebuild fails | the checkpoint row cannot be decoded; a read raises | in each case `ok: false`, one L1, empty state, batching continues, and the next start rebuilds from the fresh checkpoint |
| 42 | Thrash | a request is convicted by scans three times running for the same test | the merge-thrash ladder counts three |
| 43 | Hand-submitted | a culprit was submitted by hand | it is blocked like any other |
| 44 | Scanning off | `scan: off` | a red batch takes the fallback; no `merge_scan` row |
| 45 | Kill switch | `merge_deep.chain_cap: 0` | every request is dispatched alone; observations are still recorded |
| 46 | Rates follow the rows | 30 scanned trees at 30 s beside verifies of 30 minutes; of 50 new requests 10 were blocked or convicted, 10 landed and 30 were cleared by scans; those 30 then landed as cleared | `rates.scan_tree_cost` is near 0.017, `rates.defect_rate` near 0.18 and `rates.cleared_defect_rate` near 0.004, and the next plan uses them |
| 47 | Cleared members in a fallback | a batch of 3 new and 2 cleared requests is red and cannot be scanned; another is scanned and no member's tree fails; another times out | in each, the 3 verify alone as one red set and the 2 are new requests again |
| 48 | A landing is recorded with its kind | a cleared request lands in a passing batch | its outcome row says `landed`, kind cleared |
| 49 | A pending conviction is not batched | a culprit's conviction is pending while the merger waits for a permit | the verifier loop does not take it as a member of a batch |

## 8. Decomposition plan

Greek labels are this plan's; ids are assigned at decompose. "Signal" is what an operator observes.

| Label | Task | Touches | Signal or what it unlocks | Needs |
|---|---|---|---|---|
| **ι0** | The un-chained mark survives a restart (interim; `complexity: simple`) | `worker.py`, one test | After a restart mid-isolation the marked requests still dispatch without members | — |
| **α1** | The tested tree on the in-flight entry | new `merge_lane/tested_tree.py`; `types.py`, `verify_dispatch.py`, `dispatch_ports.py`, `worker.py`, `orchestrator/tests/_merge_lane_fakes.py` | Row 2. `get_merge_queue` shows each in-flight entry's `tree`. Unlocks α2, task 10995 | ι0 |
| **α2** | The verdict's facts reach the lane; one `merge_observation` row per verdict | new `shared/src/shared/merge_evidence.py`, new `merge_lane/observation.py`; `post_merge_verify.py`, `verify_failure_outcome.py`, `inflight_verify.py`, `dispatch_ports.py`, `types.py`, `worker.py`, `event_store.py` | On the live lane every verdict of a dispatched entry writes a `merge_observation` row with its tree, evidence class and failing checks. Rows 1, 3 (rows), 4. Unlocks σ1, ρ | α1 |
| **σ1** | The scan engine, for failing tests | new `merge_lane/culprit_scan.py`; `verify.py` (a node-id run over mapped ids that returns each module run's result; whether a tree collects a test), `git_ops.py` (the worktree kind, its protection, `reclaim_leaked`, a merge by commit), `orchestrator/scripts/warm-lane/lib_lane_state.sh`, the tests that enumerate worktree kinds, tests on real git fixtures | `python -m orchestrator.merge_lane.culprit_scan <observation id>` prints the trees' statuses and the verdict for a red batch recorded on the live lane. Rows 5 to 19, 24, 25 and 28 at the engine. Unlocks σ2 | α2 |
| **ρ** | Batch state that survives a restart: alone with its red set and the release rule, cleared, recorded outcomes with their kind; the mark becomes "alone" | new `merge_lane/batch_state.py`, `orchestrator/merge_evidence_config.py`; `config.py`, `event_store.py`, `worker.py`, `intake_buffers.py`, `intake.py`, `inflight_verify.py`, `dispatch_ports.py`, the mark's tests, a restart scene helper in `_merge_lane_fakes.py`, `plans/merge-lane-policy-prd.md` (D6) | `get_merge_queue` shows `alone` and `cleared` per waiting request; `merge_batch_state_rebuild` rows. Rows 3, 22, 23, 35, 39, 40 (alone and cleared), 41, 47, 48. From here every member of a red batch verifies alone under the release rule, which is today's behaviour plus the release. ι0's file is gone. About 14 files: one state object and its persistence are one purpose, and a split would leave state that is lost on restart | α2 |
| **σ2** | The scan in the lane: queue, supervise, apply the verdict, convict; dark_factory switched to `scan: auto` | `worker.py`, `inflight_verify.py`, `post_merge_verify.py`, `verify_failure_outcome.py`, `intake_buffers.py`, `dispatch_ports.py`, `event_store.py`, `merge_lane/batch_state.py`, `merge_lane/observation.py`, `_merge_lane_fakes.py` (the test `ScanRunner`; `lane_scene_config` pins `scan: off` unless a scene asks, since unit tests read the operational yaml), `dark-factory-orchestrator.yaml`, `plans/merge-lane-policy-prd.md` (D6), `plans/deep-merge-ahead-prd.md` (decision 9) | On the live lane a red batch is followed by a `merge_scan` row, its culprit is blocked with its own tree's failure report, and its other members show `cleared`. Rows 5 to 20, 24 to 31, 40 (scan), 42 to 44, 49 | σ1, ρ |
| **β1** | The cost rule and its rates, in shadow; `scan: auto` | new `merge_lane/batch_rule.py`, `merge_lane/rule_rates.py`; `types.py`, `worker.py`, `verify_dispatch.py`, `merge_lane/observation.py`, `shared/src/shared/merge_evidence.py` | Every `merge_observation` row carries the plan the rule would have made beside what the table dispatched, and the snapshot shows the rates with their n. Rows 32 (the switch), 33, 36, 37, 46 as function tests | σ2 |
| **σ3** | The scan also runs a red lint or type-check leg | `merge_lane/culprit_scan.py`, `verify.py` (one leg alone on a tree), `merge_lane/observation.py` (the `LEG_RED` blocker goes), tests on real git fixtures | On the live lane a batch that is red on a lint or type-check leg gets a `merge_scan` row whose checks include the leg. Row 21 | β1 |
| **β2** | The rule sizes and fills batches; cleared requests ride; `chain_policy.py` and `depth_table` are removed | `worker.py` (the placement, the reservation, `build_chain`'s contract), `verify_dispatch.py`, `merge_verify_facts.py`, `inflight_verify.py`, `dispatch_ports.py`, `chain_policy.py` (deleted), `config.py`, `defaults.yaml`, `orchestrator/tests/_merge_deep_scene.py` and the deep tests that pin a depth through the table (about ten files), `OPERATIONS.md`, `skills/orchestrate/SKILL.md`, `plans/merge-lane-policy-prd.md` (D7) | On the live lane `chain_depth_reason.basis` is `cost_rule` and batches are larger than the table's depth at the same pass estimate. Rows 32 to 34, 36 to 38, 45, 46. This leaf passes the 15-file review trigger because the table's removal reaches every test that pins a depth; task 8528, which introduced the table, touched 32 files | β1; task 10998 |
| **ε** | The readout | new `scripts/merge_evidence_readout.py` and its tests | Run on the live store it prints: verifies per landing by day; batch sizes; red batches by scan outcome and by blocker; seconds per scanned tree, scan queue waits and how long members stayed in scan; the rates with n; predicted pass against verdict in ten bins; culprits that returned unchanged and passed; cleared members later named culprit; scans that settled nothing, by outcome; checks that were absent on the root and present on the base. With no rows of a kind it prints `no <kind> rows yet`. It also prints the figures section 9 names | σ2, β1 |
| **ζ** | Integration gate | a boundary-test module over the real lane modules | Section 7 rows 1 to 49 against the real modules, and ε's output from a lane that has run σ3 and β2 | β2, σ3, ε |
| **η** | Design gate for phase 2 (a decision task; no code) | — | It escalates to Leo with ε's output and section 9 attached | β2, σ3, ε; tasks 10721, 10995, 11002; a 7-day delay after they are met |
| **θ** | Design gate: revise `plans/merge-chain-visibility-prd.md` (a decision task; no code) | — | It escalates with this PRD's 6.4 and 6.5 and the list of that PRD's decisions, contract lines, boundary rows and delivered checks that name the mark or the chain policy (`evidence/REPORT.md` section 6). Tasks 10759, 10761 and 10762 wait for it | β2 |
| **ι1** | INV-15 `scheduling-state-survives-restart` joins the checked design invariants | `docs/legibility/design-invariants.md`, `docs/legibility/design-invariants-fixtures.md`, `skills/prd/references/gates.md`, `tests/repo_guards/test_design_invariants_consistency.py` | `/prd` G7 and `/review` phase 2 walk the new invariant; the consistency guard passes with it. The invariant: in-memory state that changes which task or merge request is picked, batched or held, or how it is verified, is restored from a named store, rebuilt from named rows, or declared dropped with what happens after a restart and an owner task or a ratification; restored and rebuilt state has a test that restarts through the public seam. Its text names ι2 and ι3 as the owners of the state that exists today | — |
| **ι2** | The merge lane's register of scheduling state, and its guard | new `merge_lane/restart_state.py`; a guard test; restart scene tests where an entry has none; `docs/legibility/exception-ratifications.yaml`; a `governed_exceptions(...)` declaration of the dropped entries in the guard's test module (`shared/src/shared/governed_exceptions.py`), so that the INV-12 check sees each owner task or ratification row | The register lists every attribute the lane's stateful classes assign on themselves, every module-level mutable in `merge_lane/`, `verify_runner.py::HostAllocator`'s, and the request fields the starting inventory names, each as restored, rebuilt, dropped or not scheduling state, with its source, its restart test, or its owner task or ratification row (`evidence/REPORT.md` section 5 is the starting inventory). The guard finds those names by reading the code; a name with no entry fails the gate, naming it, and so does a restored or rebuilt entry whose restart test does not exist | β2, ι1, κ |
| **κ** | The lane's budgets, its recent-verify window and an operator's halt survive a restart (L12) | `guard_state.py` (a persistent map beside `PersistentSet`), `worker.py`, `intake_buffers.py`, `suffix_graph.py`, `merge_lane/halt.py`, `harness.py`, `OPERATIONS.md`, restart scene tests | After a restart: a request that had used two of its three conflict bounces has one left; a request that had used its retries for a landing race, a timeout or a full disk does not get them again; the recent-verify window is as before; an operator's halt of the lane is still in force, still owned by the operator, and `unhalt_merge_queue` still lifts it. Each counter keeps the lifetime it has today (the timeout and full-disk counters are per task and span submissions), and a persisted entry has an age bound (open question 5) | β2 |
| **ι3** | Design gate: the scheduler's register (a decision task; no code) | — | It escalates with ι2's register as the pattern and the inventory of `scheduler.py::Scheduler`'s state still to be made | ι2 |

Companion corrections, made by the decomposing session with Leo's word, not by a task:

- Dependencies: 5041 on β2; 9935 on ρ; 10995 on α1; 10759, 10761 and 10762 on θ in place of 10624.
- Cancelled: 10624, 8932, 10741, 10742, 10753.
- Dated notes: 7824, 7921, 8532, 9182, 10240, 10721, 10881, 10995, 10996, 11002.
- A dated delta line in `plans/priority-at-later-gates-prd.md` 6.4: no fronted suspect list exists.
- Dated notes on 9176 and 4838 (κ's operator halt).
- Leo approved these corrections on 2026-11-20 (L13). No item of ι2's starting inventory is left dropped with no
  owner (L12); ι2 writes a ratification row only for what the lane drops by design, such as reservations and
  entries in flight, which are rebuilt from the journal.

Notes for decompose:

- ι0, α1, α2, ρ, σ2, β1, β2 are one chain on `worker.py`. σ1 is parallel to ρ; σ3 is parallel to β2. ι1 is
  independent. κ and ι2 follow β2 and share `worker.py` with task 5041 through the file lock only.
- Every leaf that touches the merge-lane cluster records its raise with
  `scripts/merge_lane_metrics.py --write-baseline ... --authorize-raise <task id>`.
- σ2's change to `dark-factory-orchestrator.yaml` uses a key that ρ introduces, so the running orchestrator
  already knows it.
- Nothing here waits for tasks 9935 or 8532. Gate 10069's own edges to them are removed when it closes.
- Signal leaves: ζ, η, θ, ι2, ι3.

## 9. Phase 2: a design gate (L8)

Not designed here. η fires a week after β2, σ3 and tasks 10721, 10995 and 11002 have landed. Its subject:

| Question | What is known | What ε will show |
|---|---|---|
| The statistical design of draft 3, as the fallback and for a project that cannot scan | repriced, 7 to 14% over the release rule with a scan and 3 to 8% with none; unchanged, it collapses | red batches that ended in the fallback, by reason, and what the fallback cost |
| What an entry is built on, and a cascade that tears down only what was built on the failed entry | with large batches most are red, so what is stacked behind one is usually discarded: 18 to 21% of verify time at caps 12 and 24 in the simulation, 25 to 30% at larger caps | the share of verify time torn down, by batch size |
| Larger caps; skipping a link that does not merge instead of stopping; one going cost for both kinds from what is actually waiting | caps of 24 and 48 add 17 to 39% in the simulation; at the starting rates a batch of 12 new and 11 cleared requests costs 9 to 22% less than two batches (11% at a defect rate of 0.14), and a batch started by a new request does not form it from 0.11 (D14) | batch sizes reached; builds cut short; how many cleared requests wait |
| Expensive trees (a long build): a binary search over the build's own trees in one warm worktree; leaving out members whose changes cannot reach the failing check's build unit; two scans at a time; deciding per batch | binary search overtakes member-alone trees when a tree costs 15 to 20% of a verify; scanning stops paying at 43 to 57%; with trees at 20% one scan at a time saturates | seconds per tree; queue waits; batches sent to the fallback because the queue was full |
| A scan on the laptop; the unscoped type-check gate; files that do not collect; checks that only unlanded work brings | 56 of 104 red batches were verified on the laptop; 121 of 190 batches were stacked on unlanded work | scans where nothing reproduced, by verify host; red batches by blocker; checks absent on the root and present on the base |
| Reify and the other projects | reify's verify is one opaque script that stops at the first failing step; 51 of its 59 reds carry no ids | their red batches all take the fallback |
| Whether the lane is still the limit | the model lands 1.3 to 4 times what arrives at today's verify speed | queue depth and wait by day |

## 10. Out of scope

- The order requests are taken in (part 1). The scheduler.
- What an entry is built on (task 11002; η). The head-failure cascade.
- How verify capacity is shared with task verifies (gate 10572).
- Detecting a red main (the probe; task 7921). Holding requests while main is red (gate 10577): until then a
  batch on a red main is scanned, nobody is blamed, and its members are batched again or, the second time,
  verify alone.
- How a passed batch lands; the post-landing type check (task 10998).
- The statistical design of draft 3.
- Restoring the lane state that other tasks own: lane holds (tasks 9176 and 4838) and the journal (task 5623,
  gate 10217).

## 11. Open questions (tactical)

1. **Whether the scan stops running a member's remaining checks once it is a culprit.** Suggested: yes. Decide in
   σ1.
2. **Whether the checkpoint is one row or several.** Suggested: one. Decide in ρ.
3. **Whether member trees are started in land order or newest first.** Suggested: land order. Decide in σ1.
4. **Whether ε's bins are equal-width or equal-count.** Suggested: equal-width, n printed per bin. Decide in ε.
5. **The age bound of κ's persisted entries.** Suggested: entries of the recent-verify window older than the
   fleet's 8-hour restart clock are not loaded, since the window is reset at start today so that a restarted
   worker does not act on stale flakiness; a counter is dropped when its task is done or cancelled. Decide in κ.
