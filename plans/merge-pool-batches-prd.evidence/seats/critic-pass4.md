**Verdict: not a clean pass. 10 A-defects, 18 B. The re-phasing is sound.** No class of red batch needs draft 3's statistics from day one. Every case where draft 4 does something wrong has a deterministic fix (A1, A2, A5, A6), and the residue falls to "each member verifies alone", which is today's behaviour. One case is worse than today: B2.

Read-only against main `69972d7c6a`. Paths are under `/home/leo/src/dark-factory/orchestrator/src/orchestrator/` unless absolute. Scratch is `/tmp/gate-10069/critic4/` (`prefix_check.py`, `fset.py`, `fset2.py`, `rule.py`). Task texts and size facts come from two sub-agents; I re-verified the ones marked (v).

## A. Answer-changing

**A1. A scan conviction is weaker than a solo failure (§2 "A scan conviction is as sound as a solo verify failure … on the same host"; §6.3 `attribute`).**
- A solo red needs the full run to fail and the isolated re-run to fail. The scan convicts on one unconfirmed isolated FAIL on tree j.
- "Same host" is false under D3 for 56 of 104 red batches, which ran on the laptop.
- The flake gate filtered only the tip. `verify.py` itself warns that a loaded host starves an isolated run into a false "reproduces" (comment on `_SWEEP_PREFILTER_TIMEOUT_SECS`).
- 132 `passes_in_isolation` merge-gate rows since 10-21 show flaky ids are common.
- *Requirement:* a conviction needs a confirmed FAIL on the culprit's tree (a second run, or a stated and measured false-FAIL bound) and a PASS on the tree before. Correct the premise.

**A2. The scan tree's environment is unspecified, and a failure on the base has no bound (§6.3 "checks out each tree … `warm_seed=True`"; §6.4 "except when the reason is `base_failed`: then it is released").**
- `ephemeral_worktree` seeds `target/` only.
- `verify.py::_node_id_run` calls `run_verification` without `is_merge_verify=True`, so there is no cold pre-provision. The flake gate passes it.
- `verify.py::_provisioned_ephemeral_worktree` says a non-main-tip kind "would need its own" policy. `_provision_main_tip_worktree` records the consequence: an environment-only failure reads as red (task 6917).
- An environment or isolation-only FAIL on every tree is `base_failed`: nobody is blamed, members are released, they re-batch, and it repeats with no bound (R7). On a main base it spawns the probe each round.
- `run_scan` takes one `module_configs`, but `_group_node_ids_by_subproject` requires per-tree discovery. The owner's task-scoped set mis-maps other modules' ids (task 9922's class).
- *Requirement:* name per-tree provisioning and module discovery (merge-boundary set), and the role or admission the scan runs under (task 8882). Bound `base_failed` releases: fall back after N, or when the probe finds main green. Add rows.

**A3. `FailingChecks` describes data that does not exist (§6.2 "runnable test ids that failed in isolation on the tip"; D10).**
- Under `fails_in_isolation`, `FlakeSuppression.test_ids` is every id examined. `confirm_isolated_rerun_verdict` returns at the first failing group, so later groups are never re-run. Ids are un-grouped; mapping is per tree.
- Under `other_leg_failed` it is empty. Of the 7 such red batches, 6 had the test leg red too (`fset2.py`: lint+test 5, lint+test+type 1, type only 1). A leg-only scan then clears members whose test failures were never scanned.
- One id that passes on every tree (a flake riding along) is "unattributed", which sends every non-culprit to the fallback.
- *Requirement:* define the set from what the wire carries. Say where test ids come from when a leg is red, or make that red unscannable. Say what a check that passes on the tip's scan tree does to the other members.

**A4. `CheckStatus` cannot be produced from `run_node_ids` (§4 "today an unmapped id is an error"; §6.3).**
- Only a missing file is distinguishable. A missing function or parameter, a collection error, a setup error and a module-level id (3 of 87 sets) all read `'error'`, on the culprit's tree too.
- `'skipped'` has no mapping.
- Read as ERROR, a test added to an existing file always falls back. Read as ABSENT, a collection error is hidden (INV-11).
- *Requirement:* a detection rule and a row for each case.

**A5. `attribute` contradicts itself on a failing base (§6.3).**
- "sound when … `base_failed` is empty or every check is in it" against "When every check is in `base_failed`, every member is undetermined".
- When some checks fail on the base and the rest are attributed, non-culprits are undetermined without reason `base_failed`, so they take n solo verifies. With stacked large batches this is the common case. The simulation released them (`lane.py::_scan_done`, `base_bad`).
- *Requirement:* one truth table over (culprit, unattributed, base-failed some, base-failed all).

**A6. Applying the verdict (§6.4 "taken from its buffer (`take_buffered`) … holds prevent it").**
- The red owner goes to `_queue` (`_requeue_request`), not a lane buffer, until the merger drains it. `take_buffered` returns `None` there. Resolving it in place makes the later drain log a duplicate-submission coalesce (`intake_buffers.py::_coalesce_reentrant_drain`).
- `take_buffered` is keyed by task id, so a superseding request would be convicted.
- No tip check at apply time (INV-3).
- A link picked before the hold is not covered: reservation is 7200 s, and the longest local verify was 148.7 min.
- *Requirement:* state the owner's custody during a scan. Resolve by request id and tip. Say what a member that is not waiting at the verdict gets. Add rows for each.

**A7. The culprit's outcome cannot be built (§6.4 "the outcome a red single gets … failure category and cause hint from its own tree's failing checks").**
- `run_node_ids` returns statuses only and discards the result and the junit report. There is no failure text, category or cause hint.
- The thrash signature is `hash(category, cause_hint)`, else the normalised reason (`shared/task_metadata.py::RetryLedger.compute_merge_outcome_signature`). A reason naming batch and tree differs every time, so row 24 cannot pass.
- `spawn_main_health_probe` needs the tip `VerifyResult`. It is not in `merge_observation`, so it is lost on a re-scan after restart.
- Dropping `red_verify_outcome` for a batch also drops the flock-contention L2 alarm, and needs a non-`None` placeholder (`None` means pass).
- The call site is `post_merge_verify.py`, which is not in σ2's Touches.
- *Requirement:* the scan returns the culprit tree's evidence (text, category, hint). Name what is retained for the probe. Skip only the task-fault arm.

**A8. B3, row 16 and goal 3 contradict the §6.5 formula (`rule.py`).**
- At the starting rates a cleared candidate adds 0.23 to 12 fresh members (q 0.14), or 0.14 at q 0.05. Its going cost is 0.083.
- With `r = 0` it joins only up to 2 fresh members (0.071; 3 costs 0.099). B3 says "any batch of two or more".
- B2's configuration gives `P_red = 0`, so `sound` is 0/0.
- *Requirement:* fix the rule or the invariants. Define `P_red = 0`. Give row 16 its rates and pool order.

**A9. Decomposition (§8).**
- σ1 needs α2: `ScanInput.checks` is α2's type, and σ1's signal reads a `merge_observation` row. "Parallel after α1" does not hold.
- ρ has 9 named files plus tests, and carries rate fitting whose consumer is β and whose `merge_scan` input arrives in σ2.
- β has 13 named files plus about 10 test files. The 8528 precedent was 32 files, +2358/−1421.
- ε's scan and rate sections have no real producer at "Needs α2" (INV-13).
- σ2 omits `event_store.py` (`merge_scan`) and `post_merge_verify.py`.
- θ covers only the mark. β also breaks that PRD's `chain_policy` wire, the `basis` vocabulary and 10761's check on `chain_policy.py` (v).
- *Requirement:* corrected edges; split ρ and β; ε's sections after their producers; θ after β with the policy lines.

**A10. ι is not implementable as written (§8 ι).**
- The sites are wrong. `CONTRIBUTING.md` is pinned as an absence. The real sites are `design-invariants.md`, `design-invariants-fixtures.md` and `gates.md` (two enumerations), enforced by `/home/leo/src/dark-factory/tests/repo_guards/test_design_invariants_consistency.py::PINNED_SITES` (v).
- A guard that only checks a declaration exists fails INV-10: a "restored" entry passes whether or not it restores.
- "Changes which request is picked" is not mechanically decidable, so the guard must classify everything: 104 attributes in `SpeculativeMergeWorker.__init__`, 255 across `merge_lane`, 91 in `Scheduler`.
- "Dropped, with the task that owns it" needs live owner tasks (INV-12) that a dispatched agent cannot file.
- "Independent" is wrong: every later leaf adds attributes and would go red at the gate.
- *Requirement:* scope, discovery method, a behaviour-exercising check per restored or rebuilt entry, who files owners, ordering against the chain, and one leaf per subsystem.

## B. Minor

1. **T2 is false for gate-rebased owners.** 6 of the 60 newest red batches have a one-parent owner commit (event 619204, owner `77c7281701`). Scope T2 or make α1 need 10721.
2. **NONE evidence marks only the owner.** Today every member is marked. A hang caused by a link then recurs once per position. There were 0 timeouts in 104 red batches.
3. **D4 blocks a sound later member** whose new test fails because of an earlier culprit. The lower bound is 1 in 58 (618460). With D7 it feeds the thrash ladder.
4. **Second culprit on the same check is cleared.** Failures concentrate (`test_merge_gate_history_leg.py` is in 35 of 87 sets), so `start_cleared_defect_rate` 0.01 is unsupported. The unscoped type-check never ran on a red tree, so "passed every other check" is false.
5. **Timeouts.** `pytest.mark.timeout(1200)` overrides the injected 300 s (`pytest_timeout::_get_item_settings`). `scan_tree_budget_secs` 600 and `t = 0.05` are wrong for those sets.
6. **Hold mechanics.** `hold_for_scan(deadline)` is called before the deadline is known. There is no watchdog for a dead scan task. No merger wake-up on release (`_resume_signal`; pass-3 B3 again).
7. **Estimators.** `defect_rate` needs each member's class and the single's final disposition; neither is in the rows. Windows cannot slide over checkpoint tallies. `scan_tree_cost` does not fit events-over-trials.
8. **`EventStore.emit` returns `None`.** "Returned id" is a signature change.
9. **New `WorktreeKind`.** Also needs `orchestrator/scripts/warm-lane/lib_lane_state.sh::LANE_PROTECT_GLOB_FALLBACK` and its tests. A protected band is never reaped after process death (tasks 5584, 7458), so N3 needs a startup reclaim.
10. **Seams missing from §5:** 7919, 7053, 9132, 9268, 9269; 9129, 9134, 8882, 9922, 10240; 10881 (duplicates α1's fake move); 10741, 10742, 10753 (moot after β); 10621; 7810, 8359; 7921's own "held" mark. 11002 expects a content id on the tree; 10995 never reads it. 7922 does not need σ1.
11. **N2** needs "no check fails on the base and none errors".
12. **A member both `alone` and `cleared`:** precedence is unstated.
13. **Meaningful strings:** `unscannable`, `reason`, `because` and check keys (heuristic 12, INV-2).
14. **No streak escalation** for scans that attribute nothing (INV-4).
15. **Dead vocabulary:** `NOT_FULL_GATE` has no producer today; `log_dir` and `flake_suppressed` have no named consumer.
16. **ι0** leaves a guards file for ρ to retire. `PersistentSet` has no `.update`.
17. **β.** Selection is non-contiguous, against `build_chain`'s stated contract. The signal "reach 12" fails at a fitted q of 0.05 to 0.08, where the rule picks 11 or 10.
18. **§2 wording.**
   - The 17 unconfirmed are 10 unmapped and 7 other-leg; "largest class" is wrong.
   - Build truncation is unmeasured beyond link 5, and verify time beyond 6 members.
   - The "one red in five" row simulated draft 3's inference as the fallback. By §6.5's own formula a 12-batch costs 3.65 against 2.24 (4.24 becomes about 2.6).
   - The "draft 3" column is plain U; with base choice it was 1.64.

## Leo's call

1. **Re-phasing:** confirm, with A1, A2, A5 and A6 fixed in phase 1.
2. **Lint and type legs in phase 1.** Only 1 of 104 red batches is leg-only. Scan them (needs A3 and pyright provisioning), or send any red-leg batch to the fallback.
3. **D4 and D7.** Block every first-failer, or send "new test fails behind an earlier culprit" to verify alone and keep it off the thrash ladder.
4. **L4's price.** 5041 behind β holds 131 open dependents (sub-agent count). 9935 behind ρ delays part 1.
5. **ι.** Declaration only, or behaviour-checked; whether "dropped" is an acceptable end state.
6. **Scan priority** on the workstation against the local verify.
7. **A1's confirm run:** one extra tree per culprit.
8. **Caps 12 and 24** on extrapolated build and verify evidence.

## Checked and sound

- `_take_verdict` is synchronous and precedes `_release_chain_resources`.
- The 60 newest red tips and every prefix commit are in the local store, laptop batches included.
- `take_buffered` and `_resolve_or_drop_abandoned` are legal from a detached task under `_assert_single_writer`.
- The journal entry is removed by the result's done-callback.
- `pop_next_pickable` is the only pick.
- B1 holds at 491 grid points; the only mismatches are at exact thresholds.
- With `u = 0` the formula equals `lane.py::UPolicy._cost` and `scan_cost_table` (variant A, linear).
- §2's table and sensitivities match `results2_S_compact.txt`.
- 104, 87, 17 and 2.1 ids reproduce.
- The flake observation rides a red result and the wire.
- Recovery is awaited before the loops start.
- Companion corrections need no code.

## Not checked

- No repository test was run, and no scan was executed on a real prefix tree.
- Laptop against workstation reproducibility.
- pytest's junit output on a refused argv (taken from the `run_node_ids` docstring).
- `merge_finalized` and dashboard bookkeeping for a link blocked from its buffer.
- `REPORT.md`.
- The simulator beyond `_cost`, `scan_cost_table`, `_scan_done` and `on_ret`.
- Task texts beyond the sub-agent's quotes.
