# Evidence for `plans/merge-pool-batches-prd.md`

Gathered 2026-11-19 and 2026-11-20 in the gate 10069 session. Everything was read-only against the dark_factory
`runs.db`, syslog, the task store and git. The seats' own reports are in `seats/`, unedited; each states its
counting rules. This file says which report each of the PRD's figures comes from, which earlier claims were
withdrawn, and how the design changed.

Scripts are in `scripts/`, with the suffix `.py.txt` so that no repository scanner treats them as project code.
Rename to `.py` to run. They read `/home/leo/src/dark-factory/data/orchestrator/runs.db` read-only and wrote their
scratch under `/tmp/gate-10069/`, which is not kept. `scripts/rulecheck/rule9.py.txt` is the reference computation
of the PRD's batch rule (section 6.5, invariant B3) and runs alone.

## 1. Where each figure in the PRD's section 2 comes from

| Figure | Report | Note |
|---|---|---|
| 214 verdicts with rebuilt trees; no contradiction of "deterministic defects"; eight test files explain all 58 red batches | `seats/forensics-2-base-state.md` section 6; `seats/forensics-1-red-batches.md` | 4 reds undetermined |
| 104 red batches since 2026-10-21; 87 failed again in isolation, 2.1 tests on average; the other 17 are 10 with unmapped ids and 7 with another leg red (6 of the 7 with failing tests too) | `seats/substrate-targeted-rerun.md` section 5; `seats/critic-pass4.md` A3 and B18 | joined from `merge_verify` and `flake_occurrence` by merge sha |
| The 10 unmapped ones are batches whose owner has no module; their ids map against the full module set, 10 of 10 | `seats/critic-pass5.md` A1 | "owner has no module" is read from their `module:__global__:test` units |
| 89 of 105 red tips have every failing id present on the base; 4 a file absent; 1 a test added to an existing file | `seats/critic-pass5.md` "Checked and sound" and B3 | |
| 132 merge verifies turned green by the flake re-run | `seats/critic-pass4.md` A1 | `passes_in_isolation` merge-gate rows since 2026-10-21 |
| 56 of the 104 red batches ran on the laptop | `seats/substrate-targeted-rerun.md` section 5 | |
| Preparing a worktree's environment: 12 s at the median, 43 s at the 90th percentile | `seats/critic-pass5.md` B14 | 997 `_merge-` rows in syslog |
| A merge 1.3 s; a module's failing tests 10 to 60 s; a type-check leg 11 to 220 s | `seats/batch-build-and-conflicts.md` item 5; `seats/substrate-targeted-rerun.md` "Cost estimate" | estimates; nothing records a scanned tree yet |
| A full verify: 26 minutes on the laptop, 42 on the workstation | `seats/substrate-targeted-rerun.md` section 5 | `merge_verify` since 2026-11-06 |
| 13 of 58 red batches red only because of unlanded work ahead; 121 of 190 batches stacked | `seats/forensics-1-red-batches.md`; `seats/critic-pass6.md` B11 | |
| All 31 red-batch members re-batched before their solo verify had crossed a restart | `seats/measure.md` item 6 | |
| A link fails to merge 2% of the time (19 of 999), positions 1 to 5; build time flat for 2 to 6 members | `seats/batch-build-and-conflicts.md` items 2 and 5 | from the `build_chain:` syslog line, 2026-11-08 to 11-20 |
| Verify time against batch size: slope −0.7 minutes per member (−2.6 to +1.3), 2 to 6 members | `seats/batch-build-and-conflicts.md` item 6 | |
| Landing: 246 s per member after the first | `seats/batch-build-and-conflicts.md` item 7 | |
| Per-request defect rate 0.14 over 1,237 requests; 0.07 to 0.24 by era | `seats/predict-and-replay.md` section A | its sections B to D are superseded, see 3 below |
| One red batch in 58 failed only with two members together | `seats/forensics-2-base-state.md` section 2 | tasks 10242 and 6514 |
| The simulated table: today; no scan with the release rule; scan with every red scannable; with 16% and 5% of culprits not scannable | `seats/sim-6-fallbacks.md` section (a) ("For scale", the sticky table, column F1u) and section X1 | the 75-minute line is its W2 table; waits are from the "Headline rows" table |
| The fallback comparison (each alone, release rule, halves, repriced and unchanged statistical design) | `seats/sim-6-fallbacks.md` section (a), sticky, q 0.14, m 0.16 | |
| Concurrency inside a scan; the break-even tree cost | `seats/sim-6-fallbacks.md` sections X3 and X2 | |
| What the scan is worth before the corrections; conviction by elimination never fired | `seats/sim-5-targeted-rerun.md` | |
| The batch rule: convergence in 1,278 cells, the table as a special case, B3's numbers, the price of what happens to cleared members in a fallback | `scripts/rulecheck/rule9.py.txt`; `seats/critic-pass6.md` to `critic-pass8.md` "Checked and sound" | the critic's own grid of 4,032 cells also converged |

## 2. The simulator

`scripts/sim/lane.py.txt` is the final engine: the 8528 study's discrete-event model of the lane (2 hosts, 3
entries in the frontier, in-order landing, the head-failure cascade), with policy objects. Validation is in
`seats/sim-1-layers.md` section 1, `seats/sim-4-unified-rule.md` "Checks", `seats/sim-5-targeted-rerun.md`
section (e) and `seats/sim-6-fallbacks.md` section (e). The compact result tables are in `scripts/sim/results/`.

What the contract has that the simulator does not: members tested on the root, not the base; a cleared member of
a fallback batch becoming new again; the first kind's size computed directly; rates fitted from live rows; merge
conflicts; checks that fail only with two members together.

## 3. Claims made earlier in the session and withdrawn

- "7 of 15 fully resolved red batches had no culprit member" and "conviction by elimination would have been wrong
  in 7 of 9 cases" (`seats/predict-and-replay.md` sections B, C, D). Six of those seven batches were stacked on an
  unlanded batch that then went red; in the seventh the suspect was defective and passed after its content
  changed. `seats/forensics-1-red-batches.md` section (iii) lists the method errors.
- "The batch chooser is not worth building." It rested on a replay that held "every member verified alone" fixed.
- "Held 27 minutes on average" and "under 0.1% reach the limit" (a biased counter in the simulator).
- "A scan conviction is as sound as a solo verify failure, on the same host" (draft 4). It rested on one isolated
  failure, and the scan host differs from the verify host for 56 of 104 red batches. The PRD requires two failures
  on the member's own tree and a pass on the root and the base.
- "One red in five that cannot be scanned costs 7 to 13%" (draft 4). That run used the statistical design as the
  fallback; unchanged, that design collapses when the unscannable failure belongs to the culprit.
- "Scan a red batch when its trees cost less than its fallback would" as a per-batch test (draft 6). It has no
  fixed point for tree costs of 0.3 to 0.8 of a verify (`seats/critic-pass5.md` A2).

## 4. How the design moved

1. Drafts 1 to 3 (`draft3-statistical-design.md`): a ledger of tested trees, exact joint inference, one cost rule
   for batch membership, deferral, main as an unknown, conviction by elimination behind a self-switching
   threshold. Critic passes 1 to 3: 20, 9 and 3 answer-changing defects.
2. Leo's proposal, 2026-11-20: after a red batch, re-run the failing tests on each member; send back the ones
   that fail; return the rest at a very low failure probability. Simulated in `seats/sim-5-targeted-rerun.md`:
   conviction by elimination never fired and the gain was two to three times the statistical design's.
3. Drafts 4 to 9 and critic passes 4 to 8 (10, 3, 1, 1 and 0 answer-changing defects). The scan moved from the
   trees the batch build made (draft 4) to each member alone on the base (drafts 5 and 6) and then on the root
   (draft 7); the fallback became the release rule after `seats/sim-6-fallbacks.md`; the per-batch scan test
   became one switch per project; the rates' definitions were corrected in pass 6.
4. `seats/earlier-defects-digest.md` lists the 74 defects found in the 2026-11-13 draft of this design
   (`plans/merge-pool-batches-design-notes.md`).

## 5. Lane state lost on an orchestrator restart, and who owns it

From `seats/substrate.md` item 9 and `seats/seams-2.md` item 8. This is the starting inventory of task ι2's
register. "Changes behaviour" means the lane picks, batches, holds or verifies differently after a restart.

| State | After a restart today | Changes behaviour | Owner |
|---|---|---|---|
| The un-chained mark (`_next_dispatch_unchained`) | lost | yes: marked requests batch again | this PRD: ι0, then ρ |
| The pass-estimate window (`_chain_policy`) | rebuilt from `merge_verify` rows | no | removed by β2 |
| The bounce registry (`merge_lane/types.py::MergeBounceRegistry`, held by the conflict tracker) | lost: the cap of three starts again | yes | this PRD: κ |
| Retry counters (landing race, gate re-verify, timeout, full disk, narrowed) and chain-generation counts | lost: budgets restart | yes | this PRD: κ |
| Recent verify outcomes, probe round counter | lost, on purpose | minor | this PRD: κ, with an age bound |
| Operator halt (`merge_lane/halt.py::HaltState._operator_halt`) | lost, by design | yes | this PRD: κ |
| The main-health hold on the normal lane; the auto-heal registry | lost | yes | tasks 9176, 4838 |
| WIP halts | restored from open escalations (`Harness._rehydrate_merge_halt`) | no | — |
| Host quarantine and skew benches | lost | no longer applied (Leo, 2026-11-20; task 11000) | — |
| The redispatch park's front priority; the first-enqueue stamp | lost | yes | task 5623; gate 10217 leaf C |
| Verify-dispatch counters (the cold-verify valve, dead-verify aborts); contended-lease counters | lost | minor | none: ι2 places them |
| Reservations, look-ahead hold, permits, in-flight entries | lost; requests are re-read from the journal | no | dropped by design |

## 6. What in `plans/merge-chain-visibility-prd.md` is defined by the mark or the depth table

From `seats/seams-2.md` item 4 and `seats/critic-pass4.md` A9. Design gate θ revises these.

- Decisions D5, D6, D7, D12, D19.
- Contract 8.1: `ChainIsolationSettled`; `IsolationExit.solo_verifies`; `IsolationEnd` (`members_retired`,
  `shutdown`); `MemberExit` (`landed`, `blocked`, `dropped`, `rechained`); the isolation clause of I4; the bound
  "open isolations ≤ marked members"; `open_isolation`, `note_solo_verify`, `note_member_exit`; the end-per-exit row
  "Tip red … beside the mark"; the Isolations paragraph.
- Contract 8.2: `entries[].unchained_mark`; the top-level `chain_policy` block (`enabled`, `depth_table`,
  `pass_estimate`, `row`); `ChainDepthBasis`.
- Boundary rows H3, H4, H14, H15, H19, H21, H22, L6.
- Manifest bindings: β `unchained-mark`; β `red-tip-and-finally-sites-settled-first` (producer: task 10624);
  γ `unchained-mark-on-the-wire` and `dashboard-reads-round-view`, copied into task 10761's delivered checks as
  greps for `unchained_mark`; task 10761's check on `chain_policy.py`.
- Task texts: 10759, 10761, 10762. Task 10759 depended on task 10624.
