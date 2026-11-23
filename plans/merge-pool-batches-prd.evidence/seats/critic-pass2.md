# Critic pass 2: DRAFT 2 of `/tmp/gate-10069/draft/merge-pool-batches-prd.md`

Not a clean pass: 9 A-defects. Read-only against main `25354e0e09`. Scratch is in `/tmp/gate-10069/critic2/` (`k1.sh`, `rule_checks.py`, `root_model.py`). Code paths are relative to `/home/leo/src/dark-factory/orchestrator/src/orchestrator/`.

## A. Answer-changing

**A1. §6.4 "An observation leaves the ledger when none of its keys belongs to a live request" undoes D4, goal 4 and row 9.**
- A blocked culprit `g` is retired, so its solo red `{g}` has no live key and leaves.
- The batch red `{g,x,y}` then stands alone: `x` and `y` sit at 0.378, not 0.146. They are never released, and never convictable because `g` is gone (D5).
- The simulated policy did the opposite: `lane.py::Belief.blocked` deletes every observation the blocked request was in.
- *Requirement:* retention by evidence group, and a row-9 assertion that fails under the current sentence.

**A2. L5 is not implemented safely or completely (§6.4, §6.6, §6.7, and the §2 premise "a red main the probe misses does not convict").**
- **Heal through the lane.** A fix `F` passes on red root M, so "a pass proves its root sound". Reds recorded on M then count at full strength. Once `a` and `b` pass, `c` is 0.937 and convicted (`root_model.py`). Row 21 only covers main moving by an outside commit. The forensic window's LOOP episode was healed by a lane-landed task (10239).
- **No proved-root criterion in §6.6 "Who".** Protection rests on `main_red` alone. Its numerator counts only probe-confirmed roots. With none in 200 it is 1/217, and the last member on an unproved root is 0.913 (f = 0.011). It is 0.70 only at 0.06.
- **§6.7 and §6.4 can contradict.** A root the probe declared red is later "proved sound" by the fix's pass. Reds on it that were not voided (signature mismatch) then convict.
- **"The root's own probability is part of `p_all_sound`"** has no carrier. `p_all_sound`, `batch_cost` and `plan_batch` take no root. Folded into `p`, an unproved root cuts depth 12→6 at defect rate 0.05 and 7→3 at 0.08 (`rule_checks.py`). Conditioned out, it does nothing. Whether a shared unproved root joins evidence groups (and so the hold) is also unstated.
- **"Every landing the lane makes proves the main it creates" is false** for `GateClearance.same_tree`, `disjoint` and `history_leg` (a rebased sha lands with no verify of it), train landings, docs-lane landings and trivial passes. The ledger cannot tell these from outside commits.
- *Requirement:* settle each of the five; add the heal row; restore red-root declarations at restart (they are not in the checkpoint list).

**A3. §6.3 "built once… returned to the lane on `MergeOutcome.verdict_facts`" and D3 "one record per verify" do not match the plumbing.**
- A pass returns `None`, not a `MergeOutcome` (`post_merge_verify.py::_PostMergeVerify._pass_outcome`; `inflight_verify.py::InflightVerifyRun._take_verdict`). `on_result` exists only on the warm path and fires before the cross-check. A trivial pass therefore reaches the lane as a pass, clears keys and proves a root, breaking V1.
- `merge_verify` is written per attempt in `verify_runner.py::VerifyRunnerPool.dispatch`. `merge_lane/shadow.py` writes rows through it too.
- The cross-check's local run writes no row (`verify_cross_check.py`). `verify_cross_check_remote_green` defaults to True (df's yaml sets false). A remote PASS row can stand beside a lane RED.
- The low-disk skip returns before any dispatch, so `SKIPPED` has no row (row 4).
- `verdict_facts(result)` cannot yield `NOT_FULL_GATE` or `BASE_UNRESOLVED`.
- The live ledger is fed by the lane; `rebuild` is fed by rows. A verify cancelled after its row was written also splits them, so I3 fails.
- *Requirement:* one producer of an observation; a named carrier for pass facts; a rule for rows with no tree and for overridden rows.

**A4. §6.8 restart.**
- `EventStore` has no "rows after id" read. `fetch_recent_events_by_type` returns `[]` on error, so "the read fails" looks like "first start" and the lane batches on an empty ledger.
- A `merge_verify` row written before a checkpoint but recorded after it is in neither checkpoint nor tail. The cross-check or red classification sits between emit and record.
- `rebuild(checkpoint, tail)` has no live set. Nothing retires requests that vanished across the restart.
- "Batching off" has no exit. The same undecodable checkpoint is the newest at every later start.
- Row 24 "same holds" is unsatisfiable: the verify a hold waited for died with the process.
- *Requirement:* a raising, bounded read (`event_store.py` is in no leaf's files); a watermark or overlap with de-duplication by `obs_id`; an exit from the degraded state.

**A5. §6.1 identity.** Results from `k1.sh`:
- **K1 is false for default-context ids.** A clean rebase where main changed a line two below the hunk gives `050fae22…` → `535246fb…`. A clean unstack whose base touched the context gives `b90b2f48…` → `5ad76351…`. `plans/one-merge-request-per-task-prd.md` D6 itself says shifted context changes the id.
- **"Newest merge commit on the tip's first-parent path" is the wrong cut** when the branch merged main into itself. Two branches with different pre-merge commits both get `a71c617b…`. It also switches off D6's "merge commit in range ⇒ DIFFERENT" arm in `classify_content`.
- K1 does hold for a far-line rebase and for the bounce onto a frozen tip of lane merges.
- *Requirement:* restate K1 with its real limit, or use context-free ids. Cut only at a commit the lane recorded (the bounce can write the stack ledger). State what D8 and the conviction truth lose when an unchanged return gets a new id.

**A6. §6.5 Holding and D7 are not what was simulated.**
- **Clock.** In `lane.py::UPolicy.next_entry` the stamp is cleared whenever the group has no test in flight. "U w0 no deadline" equals "U w0" in all five cells: the limit never bound. The draft's `held_since`, checkpointed start times and "held at most twice the p95" read as cumulative. A cumulative clock releases the later members of any set needing more than two verifies.
- **Trigger.** The simulation holds a group when any key is in a running tree, as built-on or reserved link. The draft says "a key of a request that has been picked". Reserved links and trees stacked on a vacated red batch fall outside it.
- **Pick-ahead.** The simulation decides when a host is free. The lane decides at the merger's pick, so the "no other request can be picked" fallback fires early.
- *Requirement:* define start, reset, trigger and where the fallback is evaluated. R7's bound then needs Leo's word (below).

**A7. §6.6 switch and truth.**
- "The 10 newest shadow truths" is a sliding window. At precision 0.7 it turns ON within 50 truths with probability 0.82; at 0.6, 0.46 (`root_model.py`).
- The truth definition is met by the convicting observation itself. It needs "recorded after the conviction row".
- ON-state truths exist only for requests that return with the same content id. With wrong share w and unchanged-return share r, a window of 20 suspends with probability 0.17 at w = 0.3, r = 0.5 and 0.04 at w = 0.2, r = 0.5. A5 lowers r. Nobody measures r. "6.6's suspension is the guard" for the low-biased `false_failure` is therefore unsupported.
- Criterion (d) excludes every journal-recovered request, which is the whole pool after each restart. The draft still says `RecoveredMergeOwner._file_l1` files convictions and that task 10216 gains `REJOIN_ON` membership.
- `OFF → auto` needs a config edge the rows cannot show.
- *Requirement:* window semantics, truth timing, r reported by ε, and reconciling (d) with the recovered-owner and 10216 notes.

**A8. §8 decomposition.**
- **β2 touches about 27 files**: 14 source and doc, 13 test. It also needs `merge_verify_facts.py`, `types.py`, `OPERATIONS.md` and `skills/orchestrate/SKILL.md`, none listed.
- **β2 has no test seam for batch size.** Every deep scene pins depth with `depth_table=[DepthRow(0.0, chain_cap)]` (`orchestrator/tests/_merge_deep_scene.py`; about 12,600 lines of deep tests). The rule gives only 3, 6, 7 or 12 for uniform fresh requests. Starting rates are not configurable, and D11 forbids a table mode.
- `event_store.py` is missing from β1a, γ and δ, which add six event types between them.
- γ and β2 both edit `worker.py` with no edge between them.
- η needs ε but does not list it. "7 days of rows" is not a dependency; it should be a milestone delay.
- ε's "hold times" has no producer row: `plan.held` carries no times.
- Revision 5 gives task 10759 "released" and "resolved" exits, but `EvidenceLedger` exposes nothing 10759 can read them from. Its ledger is in-memory while "a restart ends nothing".
- *Requirement:* split β2 and give it a deterministic sizing seam; declare the files; add the edges and the ledger read 10759 needs.

**A9. §6.5 "`predicted_pass` is `1 − P_red`".**
- It is conditional on the built-on entries being sound. The verdict is the whole tree's. The simulation recorded `(1−e)·P(tree sound)` (`lane.py::JPolicy._entry`).
- In β1b's shadow it belongs to the rule's plan, not the batch the table dispatched.
- ε's calibration and η's gate then compare unlike things.
- *Requirement:* record the probability of the dispatched tree's verdict.

## Contract against the simulated policy "U" (w=0, strict)

Formulas that match `lane.py`: `going_cost` (`a(0)`, the recursion, the k=1 term, k ≤ `max_fresh`), `_cost`, the join test, caps 12 and 24, and the strict rule.

| Difference | Material? |
|---|---|
| Rates: simulation used the true defect rate and fixed e = 0.009. Contract fits three rates online. The one adaptive variant moved U by −3.4% and +10.4% | yes, unsimulated |
| Main unknown, proved roots, `main_red` | yes (A2) |
| Hold clock, trigger, pick-ahead | yes (A6) |
| Blocked request: simulation deletes its observations | yes (A1) |
| Conviction reach: simulation convicts any waiting or parked request | yes (A7) |
| `predicted_pass` | yes (A9) |
| "A cleared request always joins": by the formula it does not join a lone fresh front at 0.14 (cost increase 0.176 against a(0) 0.086) | small |
| Fresh cap: simulation skips on past the cap to collect riders; contract "stops" | small |
| Group bound: simulation caps at 16 variables with shedding; contract uses class counts and `max_states` | at low defect rates only |
| No-evidence guard, parked items, pending groups, `given_sound` (an alias of the ratio) | small, unsimulated |
| Grid: simulation interpolates, and treats a same-cell posterior as cost 1 | small |

## B. Minor

1. "Per-observation probability" for an over-`max_states` group is undefined. States should factor given the roots: ten disjoint reds on one unproved root exceed 65,536 by the stated product.
2. `InFlightView` is undefined. B4 calls `is_held` pure, yet it needs a clock owner.
3. The hold also demotes high-lane requests, such as a main-fix, below normal ones.
4. "Held" collides with task 7921's `held` in the same pick filter. The simulation's word is "deferred".
5. "Already built… the rule applies to its members" contradicts task 10721 for a rebased head.
6. D13 "stacked requests make no observation": once unstacked they are ordinary fronts.
7. §6.3 "No existing key changes meaning", but §6.5 reshapes `chain_depth_reason`.
8. INV-4: `base_unresolved`, content-id `None` and skipped rows have no rate threshold. A base-resolution bug would silently remove all red-batch isolation.
9. G1: `flake_suppressed`, `merge_evidence_rebuild` and `given_sound` name no consumer.
10. Nobody refreshes identity after the lane's own bounce (`suffix_graph.py` is in no leaf).
11. Recovery-time content ids are an unbounded sequential git fan-out before the first pick (INV-8).
12. Shadow rows need one per key.
13. α2's signal says "every row". It means rows of dispatched entries.
14. `plans/merge-pool-batches-prd.evidence/` does not exist, and no step commits `REPORT.md` or the simulator from `/tmp`.
15. Gate 10069 itself depends on 9935 and 8532, while 9935 is to depend on β2. Run the cycle check at decompose.
16. Draft 1's L8 (lane state restored or listed as dropped) is gone from §3.
17. "5 of 20" and "30 days" have no stated basis (R10).

## Status of pass-1 defects

| | Status |
|---|---|
| A1 | answered |
| A2 | answered; tree-less rows need an explicit reader rule |
| A3 | partly: mapping present; pass carrier, SKIPPED and context-derived values remain (new A3) |
| A4 | answered |
| A5 | answered for reds |
| A6 | partly: reserved-link wording (new A6) |
| A7 | partly: K1 and the cut (new A5) |
| A8 | partly: root-or-member match is in; red-versus-proved contradiction remains (new A2) |
| A9 | partly: windows defined; retention (new A1) |
| A10 | answered by L3 |
| A11 | answered; B1 above |
| A12 | partly: heal case and steady-state prior (new A2) |
| A13 | partly: rows and hysteresis are in; window and truth timing remain (new A7) |
| A14 | answered |
| A15 | partly (new A4) |
| A16 | partly: no queue jump; clock and Leo's word remain |
| A17 | answered, except the recovered-owner contradiction |
| A18 | partly: 10759's read API; Revision 5 is a redesign |
| A19 | removed; see B16 |
| A20 | α and β1 split; β2 now oversized; γ's edge still missing |

B1, B2, B3, B6, B7, B9, B10, B13, B14: answered. B4: partly. B5: partly (two consumers still unnamed). B8: not answered (a link whose reservation expired is waiting and unpicked while its tree is unreturned). B11: partly. B12: not answered.

Earlier-draft items:
- Pass 1 A1, A14 and A15: answered. Pass 1 A6.5: moot.
- Pass 2 A8.2: answered, with A5's cut. Pass 2 A8 "no id": answered.
- Pass 2 A10: partly. A tree on an older unproved root is covered only by the root prior (A2).
- Pass 2 A11 and B10: answered. Pass 2 B15: partly.

## Gates

- **G1:** B9.
- **G2/G6:** A9; ε's hold times; η's wait.
- **G3:** A3; A4's read; β2's sizing seam. Other anchors hold.
- **G4:** 10759's API; 10721 (B5); 7921's name (B4). The 10217 names (`update_waiting`, `ChainLink`, `REJOIN_ON`) exist in that PRD. 10995's use of the tree fits its item (b).
- **G5:** root carrier, hold clock, retention, truth timing.
- **G7 hits:**
  - INV-4: B8.
  - INV-5/9: two observation producers.
  - INV-7: batching-off exit.
  - INV-10: row 24.
  - INV-11: failed read.
  - INV-13: hold times.
  - INV-8: advisory only.
- **META, still open:** what a pass proves on a root that already carries reds; whether conviction requires a proved root; the hold's bound; what guards ON without a control sample.

## Section 2 figures stronger than their evidence

- **Row 1 of the table** (+18/+15/+13/+11) is the upper of two calibrations. The other gives +10/+8/+8/+6 (`results2_X_compact.txt`; task 11002 states both). Rows 2 and 3 are products with it.
- **D6 "27 minutes"** is `dw_sum/dw_n`. That counts only dispatches where a stale hold stamp survived, not the mean hold.
- **D7 "under 0.1%"** comes from the same sample, and the limit never bound (A6).
- **"0 wrong in 2,556"** is at e = 0 only. At e = 0.01 strict and plain convict wrongly at the same rate: 0.28 per 100 landings at 0.14.
- **"Stays near 0.70"** is analytic at `main_red` 0.06. No simulation has a root unknown.

## Leo's call

1. **R7 and D7.** Is the bound per test in flight (as simulated; a set of n waits up to n−1 verifies) or cumulative (bounded, less elimination)? Either way D7 amends gate 10069's "verified alone ahead of every other pick".
2. **L5 and healing through the lane.** Options: accept, with shadow as the guard; a root with reds from two or more unrelated trees is not proved by one pass; conviction requires a root proved before the red.
3. **L7 window.** First 10, non-overlapping tens, or sliding.
4. **ON without a control sample.** Options: accept with ε reporting r; a stricter trigger (2 of 20); periodic re-qualification in shadow.
5. **Conviction reach.** Is workflow-awaited only intended, given it excludes the recovered pool after each restart, or should it wait for 10216?
6. **Hold and the high lane.** Does the hold apply there?
7. **Identity.** Keep task 10215's context-sensitive ids, or move both PRDs to context-free ones.
8. **Draft 1's L8.** Withdrawn, or lost in the rewrite?

## Checked and sound

- B1: 18 of 20 at e = 0.009, 20 of 20 at e ≥ 0.018, 16 of 20 at e = 0.
- B2 in spot cases (3 and 12 members at 0.14, 12 at 0.03).
- I4 and all worked values, including 0.70.
- Thresholds 0.19, 0.018 and 0.0097; starting rates 1/38 and 1/17.
- +12/+8/+5/0/+1 and "7 in 485" match `results2_Ua_compact.txt` and `out_checksU.txt`. Cap-binding shares match.
- After an ordinary landing main is the member's merge commit. After a batch it is the chain tip, reached link by link. A re-verified rebased head carries the rebased sha as its merge commit (`rebased_head.py::rebased_head_item`).
- `lookahead_hold.py` makes successors stack on the chain tip.
- V2 is feasible for reds: `_take_verdict` is synchronous and precedes the release.
- Clearing `_last_merge_block_reason` skips the thrash ladder.
- These exist: `_link_refusal`, `_default_coalesce_exclusion_reason`, `RecoveredMergeOwner._file_l1`, `foreign_commit_cut`, `own_delta_upstream`, `publish_verify_base`, `MergeRequest.awaited_by_workflow`, task 11002.

## Not checked

- `REPORT.md` (absent).
- The measured figures were not recomputed from `ev.jsonl`.
- Simulator internals beyond policy, hold, cascade and main-red paths.
- `Ud` and `Ue` were skimmed only.
- Whether cause hints are stable enough for signature voids.
- The exact hook sites for `note_in_flight` in `_merger_loop`.
- Task texts 9852, 7810, 6661 and 10762.
- Reify's config.
- No repository tests were run.
