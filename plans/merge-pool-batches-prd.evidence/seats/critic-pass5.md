**Verdict: not a clean pass. 3 A-defects, 22 B.** For dark-factory's own rates (tree cost about 0.05) the scan, lane and rule hold together; the A-defects are one unimplemented ruling (L10), one ill-posed rule (`scan: auto`) and one wrong tree rule (6.1).

Read-only against main `9856ed15dc`. Code paths are under `/home/leo/src/dark-factory/orchestrator/src/orchestrator/` unless absolute. Scratch is in `/tmp/gate-10069/critic5/` (`unmapped.py`, `kinds.py`, `newtest.py`, `basekind.py`, `rulechk2.py`, `fp2.py`, `fp3.py`, `order.py`). Task facts come from one sub-agent's sqlite read and were not re-verified.

## 1. Pass-4 defects against DRAFT 6

| Pass 4 | Status | What remains |
|---|---|---|
| A1 | answered | — |
| A2 | partly | module set is wrong for zero-module owners (new A1) |
| A3 | answered | global-chain completeness (new A1) |
| A4 | partly | "file present, id names no test" is still `ERROR` (B3) |
| A5 | answered | — |
| A6 | answered | — |
| A7 | answered | unscoped-gate arm (B12); `log_dir` (B21) |
| A8 | answered | B3's numbers reproduce; `P = 0` is defined |
| A9 | partly | ρ still large; σ3 and β1 share a file with no edge (B16) |
| A10 | partly | ι1's sites are right; INV-12 home and discovery holes (B17) |
| B1–B6, B8, B11, B12, B14, B16, B17 | answered | `guard_state.py::PersistentSet.update` exists; β2 must still change `worker.py::build_chain`'s "contiguous prefix" text and the `snapshot[:d - 1]` reservation |
| B7 | partly | member kind only exists from β1; voided red singles count (B7) |
| B9 | partly | Touches (B13) |
| B10 | answered, except 10995 | its text never reads the tree, and no dated note is listed |
| B13 | partly | check keys mix test ids and unit names in one string |
| B15 | partly | `FailureEvidence.log_dir` has no producer or consumer |
| B18 | partly | column 2 is still draft 3's rule (B19) |

## 2. New defects

### A. Answer-changing

**A1. L10's first class is not built (6.3 "the module set the batch's verify spec was built from"; §4 same row; §5 the 7616 row; row 16).**
- The flake-unmapped red batches are zero-module-owner batches. Of the 11 unmapped red tips since 10-21, 10 carry only `module:__global__:test` units (`unmapped.py`; the 11th is nested inner ids). "Zero-module owner" is inferred from those units, not from task records.
- For such an owner the spec's module set is empty: `merge_lane/post_merge_verify.py::merge_boundary_module_configs` never widens `[]`. Under 6.3 every id then "maps to no module", is `ERROR`, and the batch takes the fallback.
- Row 16 asserts the opposite ("the ids map with the full module set"). They do map against the registry: 10 of 10.
- The 7616 row says such a red "has no readable ids". It has: the flake observation names them.
- That verify is the global `&&` chain (`/home/leo/src/dark-factory/dark-factory-orchestrator.yaml` `test_command`), which stops at the first failing module, so the failing set is incomplete.
- `start_unscannable_share` 0.07 assumes these 10 scan.
- *Requirement:* name the module set the scan maps with when the spec's set is empty; state that a short-circuiting global leg is not `complete` (convict, never clear); correct the 7616 row and the starting share; or rule these batches out of L10.

**A2. `scan: auto` is ill-posed (6.5 "Whether a red batch is scanned", "Going costs", "Members"; D6; B3; row 28).** Dark-factory at t ≈ 0.05 is unaffected; `on` and `off` converge in all 3,770 cells tried.
- *No fixed point.* `F(n)` contains `A_new`, and the predicate reads `F(n)`. The reference iteration 2-cycles; at q 0.14, u 0.07, t 0.48–0.60 it alternates 0.166 and 0.501 (`fp2.py`). The reference returns whichever iterate is the 200th, and B3 pins that.

  | q (u 0.07) | cycles for t in |
  |---|---|
  | 0.05 | 0.48–0.55 |
  | 0.14 | 0.48–0.60 |
  | 0.20 | 0.42–0.70 |

  Over the wider grid (q to 0.5, u to 1) cells fail from t = 0.30 to 0.80 (`fp3.py`).
- *The walk stalls at one member.* From t ≈ 0.39 a 2-batch is not scanned, so the second member costs more than `A_new` and never joins. At q 0.14, t 0.40 (fixed point intact) the walk gives f = 1 at 1.00 verifies per landing; the optimum is f = 12 at 0.152, and `scan: off` gives 0.633 (`rulechk2.py`). Row 28 only tests t = 0.8.
- *The cited evidence is a different rule.* "35 of 36 cells" (sim-6 X2) ran fallbacks F1, F2 and F3b, never L9's. The simulator's predicate compares trees against n, not `F(n)`: `/tmp/gate-10069/sim/lane.py::FBCost.unscanned` calls `fallback(k, p, 1.0, {})`. It also adds members jointly (`UPolicy._compose`, `lookahead`), which sim-6's [RANK] note said the contract needs.
- *Requirement:* a predicate with a unique solution over the validated range, with the solver's start, tolerance and non-convergence behaviour stated; a membership rule that cannot stop below a size the cost prefers; boundary rows where scanning pays only for larger batches; a range for `scan_tree_cost`. Otherwise ship `on` and `off` and leave `auto` to η.

**A3. The base rule mislabels live stacks (6.1 "the tree of the live in-flight entry whose successor base equals `base_commit` … `None` when no live entry owns it"; row 2).**
- The lane treats an entry's own merge commit as a base too: `merge_lane/remerge_base.py::entry_bases`, `merge_lane/worker.py::_frontier_base_commits`, `merge_verify_facts.py::classify_base_relation`.
- An item merged by look-ahead onto owner A before A's dispatch chained links has base = A's own commit, not A's tip. 12 of 107 stacked batch dispatches since 11-12 are this case (`basekind.py`, first-parent arithmetic on `merge_verify` rows).
- Read literally, 6.1 gives them `base_commit = None`, so `BASE_UNRESOLVED`, evidence `NONE`, and every member alone, on a pass as well. Read loosely, built-on members come out empty though the tree contains A's owner, which 10995 and 11002 would read.
- 6.4's `RED` row also lists "the base unknown", which 6.2 already makes `NONE`.
- *Requirement:* resolve a base that is a live entry's own commit to that entry's owner and its built-on members, not its links; pick one home for "base unknown".

### B. Minor

1. **`auto` in σ2.** 6.4's table needs `worth_scanning` and the rates, which β1 delivers; row 28 is β2's. State that `auto` acts as `on` until β2.
2. **ρ before σ2.** What ρ does with a `RED` batch is unstated (presumably the fallback). It changes live behaviour under the old table; say so.
3. **`ALONE` carries no red set.** In rows 7, 9 and 21 with no culprit found, the likeliest culprit sits outside the others' red set, so nothing releases them: n solo verifies, more than `F(n)`. A test function added to an existing file is `ERROR` on the base and on every other member, so all go `ALONE` (1 of 105 tips, `newtest.py`). Share the scan's red set; define that status.
4. **Cleared members in the fallback.** The contract sends them alone (6.4; `F(n)` over f + c). The simulation kept them cleared and priced them at a0 (`lane.py::_red_no_scan`, `_cost`). §2's figures are the simulated variant.
5. **Order dependence.** At q 0.14 a new-front walk yields (12,0) then (0,11) at 2.94 verifies; the joint batch costs 2.68 (9% less; 17% at q 0.08). At u 0.16 the refusal is right. The cleared-front case passes by 0.151 against 0.152. `A_cleared` assumes 24 cleared are waiting (`order.py`). This is a property, not a defect; record it.
6. **Unlanded bases.** D1's "depends on nothing but itself" holds only on a landed base. The simulation held such reds until the entries ahead resolved and touched nobody if one failed; the contract convicts or falls back at once.
7. **Estimators.**
   - Red singles are written at the verdict, so cascade-voided reds count as defects.
   - `u` is one constant per batch; the simulated model grows with size (20–35% at 12) and picked k = 10 where the contract picks 12.
   - The 0.07 start omits module-level ids (3), nested ids (1) and laptop reds that will not reproduce.
8. **`auto` can latch off.** `scan_tree_cost` is only measured when scans run. Name what re-measures it and whether over-budget trees count.
9. **Ambiguous mapping.** 19 relative test paths exist in more than one module, covering 5 of 215 failing ids; `verify.py::_group_node_ids_by_subproject` takes the first. Disambiguate by the red test units, or treat as `ERROR`.
10. **A `RED` with no checks and no blocker** (`merge_no_evidence`; `units` absent with a red leg) is scanned with nothing to run. Give it a blocker.
11. **`note_blocked`'s trigger.** `_finalize_inflight`'s fail arm also delivers timeouts, skips and infra holds. Define the trigger by the single's recorded `RED`, kept on the entry until finalize.
12. **The unscoped-gate arm** still spawns the owner's dry run for a batch (`merge_lane/verify_failure_outcome.py::_unscoped_gate_outcome`). `RedVerifyContext` has no member count.
13. **σ1 Touches.** The startup-reclaim file is missing, as are the tests that pin kinds. `_mainprobe-*` directories from 07-17 and 09-12 are still in `.worktrees/`, so the leak is real today.
14. **Tree budget.** Say whether it covers provisioning (median 11.7 s, p90 42.7 s, max 193 s over 997 `_merge-` rows in syslog), the confirming run and its wait. Provisioning must follow the merge.
15. **Scan seam for lane tests.** No port is named; `_merge_lane_fakes.py` is not in σ2's Touches.
16. **Sizes and edges.** ρ has 8 source files, at least 5 test files and a plan, with no over-trigger note. ζ carries 41 rows. σ3 and β1 both edit `observation.py`; the overlay requires a real edge.
17. **ι.**
    - Ratifications must be rows in `/home/leo/src/dark-factory/docs/legibility/exception-ratifications.yaml`, not PRD section 3. That file and the test-tree declaration are missing from ι2's Touches.
    - The discovery rule misses state on value objects (5623's field, which is in the starting inventory), lazily set attributes, module caches and `verify_runner.py::HostAllocator`.
    - ι1 lands a checked invariant that the lane and scheduler violate until ι2 and ι3. It names no slug.
18. **Persistence.** No wire form for a pending `MergeOutcome`. A crash between `scan_ended` and the verdict rows double-counts `note_base_failed` on the re-run. "Observation id" and the CLI's "row id" are different ids, and `EventStore` has no read by id.
19. **§2 wording.**
    - Column 2 is draft 3's rule. L9 with no scan gives 2.73, 2.02, 1.35, 0.97.
    - "The fifth column lands 6.4 …" are the 16% cells; the slow-verify world was run at 16% only.
    - The 10-hour waits are fast-world figures.
20. **Reify (chain cap 6, live).** At β2 its batches are sized at 12 on u = 0.07 while every red is unscannable: 0.735 verifies per landing against 0.633. It had 5 red batches in a month, so the share never learns. Set `scan: off` there. Say whether σ3's leg scan applies to it.
21. **Small.** 8532's window straddles ι0 and ρ. Evidence path roots differ (`plans/….evidence/` against `evidence/`), and staging has no `scripts/sim/`. Goal 1's "at once" is really "before the merger's next pick".
22. **Unnamed paths.** The suffix bounce is not listed among "in scan" exclusions; it is covered only if "in scan" rides `is_reserved`. A trivial pass that adopts is `NONE` and so "fallback" by the table.

## 3. Leo's call

1. **Zero-module-owner batches (A1).** Scan with the registry, convicting but never clearing; or leave them to the fallback until 7616, starting u near 0.2.
2. **`auto` (A2).** Compare trees against n, as simulated; or keep `F(n)` with a defined solver and joint membership; or ship `on` and `off` only. The last is enough at t ≈ 0.05.
3. **Cleared members in a fallback (B4).** Alone, as written; or still cleared, as simulated.
4. **A red on an unlanded base (B6).** Act at once, or hold until the entries ahead resolve.
5. **Reify at β2 (B20).** `scan: off`.
6. **ρ (B16).** Split state from persistence, or accept the size.

## 4. Checked and sound

- **Verdict path.** `_take_verdict` is synchronous and precedes `_release_chain_resources`.
- **Four trees at once.** The merge role takes no admission slot (`/home/leo/src/dark-factory/shared/src/shared/verify_admission.py::_GATED_ROLES`), `cpu_governance` is unset, and no lock spans worktrees.
- **Failing ids.** `test_output` is whole, joined across modules, and rides the wire. The flake observation's ids are empty exactly under `other_leg_failed`, no ids and `merge_no_evidence`. `ABSENT` is computable from a mapped id.
- **Conviction from a buffer.** `LANE_BUFFERED → TERMINAL` is legal. `merge_finalized` and the journal removal both ride done-callbacks (`merge_lane/intake.py`, `merge_lane/intake_buffers.py`). Request ids survive recovery.
- **`note_blocked` placement.** A red single's block is only delivered at the head, because the cascade voids reds behind a failed entry.
- **Event store.** `emit` commits synchronously; the three new methods are absent, as §4 says.
- **The rule.** B1 (4 of 246, at thresholds), B2 and B3 reproduce. Three hand cases agree: C(12,0) = 1.828, `A_cleared` = 0.062, and the 0.210 > 0.155 refusal.
- **§2.** Columns 1, 3, 4 and 5, the fallback bullet, the concurrency bullet and L8's 7–14% match sim-6.
- **History.** 89 of 105 red tips have every failing id present on the base; 4 have the file absent.
- **Reify.** Its reds carry no ids (51 of 59 `no_recoverable_node_ids`).
- **ι1.** Its site list matches `PINNED_SITES`; the guard pins no count.

## 5. Not checked

- No repository test was run and no scan was executed on a real tree.
- The forensic figures (214 verdicts, 13 of 58, 31) and the caps figure (17–39%) were not recomputed.
- The simulator beyond `FBCost`, `_cost`, `_compose`, `_red_no_scan` and `want_scan`; no rerun.
- Whether the workstation reproduces laptop reds.
- The frozen-prefix invariant checkers against non-adjacent members.
- Config guards for a new section; dashboard readers of the snapshot.
- pytest's exit for a missing id, and quoting of parametrized ids.
- Leaf sizes are estimates.
