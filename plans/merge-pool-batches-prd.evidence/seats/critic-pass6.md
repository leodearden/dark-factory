**Verdict: not a clean pass. 1 A-defect, 12 B.** The scan, the tree rule and the rule's solver now hold; the A-defect is in how the rates are recorded and defined (new in draft 7).

Read-only against main `9856ed15dc`. Code paths are under `/home/leo/src/dark-factory/orchestrator/src/orchestrator/` unless absolute. Scratch is `/tmp/gate-10069/critic6/` (`rule7.py`, `chk1.py`–`chk6.py`, `staleroot2.py`). No repository test was run.

## 1. Pass-5 items against DRAFT 7

| Pass 5 | Status | What remains |
|---|---|---|
| A1 | answered | — |
| A2 | answered | the second kind can still stop short (B2) |
| A3 | answered | — |
| B1, B2, B3, B6, B9–B12, B14–B16, B18–B20, B22 | answered | — |
| B4 | partly | "under 1%" is 5–8% (B1) |
| B5 | answered | recorded in D14; section 9's figure is stale (B10) |
| B7 | partly | voided singles are excluded; the new definitions are wrong (A1) |
| B8 | answered | the threshold cannot be reached (B5) |
| B13 | answered | `reclaim_leaked` has no caller (B9) |
| B17 | partly | no INV-12 declaration (B8) |
| B21 | partly | `REPORT.md` and the staged scripts are behind the draft (B10) |

## 2. New defects

### A. Answer-changing

**A1. The fitted rates are not the rule's `q` and `r` (6.4 "`note_landed` is called from the request's done-callback … Each writes the request's kind at that moment"; 6.5 rate table; section 5, 9182 row "ρ calls `forget` in `_retire_item`").**

- *As written, every landing is recorded as new.* `merge_lane/worker.py::_resolve_or_drop_abandoned` sets the result and calls `_retire_item` in the same step; asyncio runs done-callbacks afterwards (checked in scratch). `forget` has dropped the cleared tip before `note_landed` reads the kind.
  - Cleared trials then contain failures only: r̂ = (e + 0.2) / (e + 20), about 0.16 to 0.27 in a 600-row window (assuming 60% of outcomes are cleared and a true r of 0.01–0.02).
  - At r̂ 0.10–0.20 the rule picks 7 to 9 new members instead of 12: 23 to 42% more verifies per landing at the true rates (`chk6.py`).
- *If the kind is captured first, `defect_rate` is biased.* A sound new request in a red batch is cleared and later lands as cleared, so it never enters the denominator. q̂ ≈ q / (q + (1 − q)^f): 0.39 at a true 0.14 with batches of 12, 0.63 at 0.20 (`chk4.py`).
  - While scanning is on the plan stays at 12, but `pass_estimate` and ε's calibration are wrong.
  - When scanning goes off, the rule sizes with q̂: 1 member instead of 4, 58% more verifies per landing, until about 600 outcome rows turn over.
- Row 46 passes under both.
- *Requirement:* write outcome rows where the outcome is delivered, before `forget`. Count a new request that a scan found sound as a sound trial of `defect_rate`, or define `q` per request. Add boundary rows for a cleared request that lands and for a new request cleared by a scan.

### B. Minor

1. **D13's price is understated ("Premises checked": "under 1% by the rule's own arithmetic").** With cleared members of a fallback batch verifying alone, `rule7.py` gives `A_cleared` 40% higher and `A_new` 5 to 8% higher at the starting rates; 15 to 17% at r 0.02, u 0.3 (`chk5.py`). The simulated table kept them cleared.
2. **The second kind still joins one at a time.** Cleared front, 6 cleared waiting, q 0.20: the rule gives (0, 6); then 12 new go separately at 3.39 verifies, against 3.06 for (12, 6). The first new member's marginal cost is 0.204 against `A_new` 0.195; later ones are cheaper (`chk3.py`). `rule7.py`'s stall check covers the first kind only.
3. **A pending conviction is still a batch candidate.** The merger waits on `SpeculationController.acquire_for_lookahead` between picks, while the verifier loop builds batches from `chain_snapshot`. Hide such a request as "in scan" is hidden; name both pick sites (`_acquire_next_request` and `_look_ahead`).
4. **No bound on repeated releases.** A member whose own tree is unsure every time (will not merge alone on the root, or flaky) is `RELEASED` each scan. Only `CONTROL_FAILED` counts. R7 needs a count over any release.
5. **`scan: auto` cannot switch off in dark_factory.** A tree is capped at 600 s and a passed verify takes 26 to 42 minutes, so the ratio stays under 0.38 against the 0.4 threshold. Say whether tree seconds include provisioning, or that this is intended.
6. **`unscannable_share`'s denominator is ambiguous.** "Blocked from scanning" may or may not include red batches while scanning is off, before σ2, or sent away as `SCAN_BUSY`. If they count, u starts near 1 when a project turns scanning on.
7. **σ2's yaml change reaches every orchestrator unit test.** `/home/leo/src/dark-factory/orchestrator/tests/conftest.py::_isolate_orch_config` binds the operational yaml, and `_merge_lane_fakes.py::lane_scene_config` passes no `merge_evidence`. After σ2 every lane scene scans, with the production runner, unless the helper pins `scan`. The running orchestrator is safe either way: unknown keys are ignored with a census warning. The laptop is expected to read its own copy (not re-checked this pass).
8. **ι2's dropped entries are an INV-12 list with no declaration.** They need a `governed_exceptions(...)` call in a test module (`/home/leo/src/dark-factory/shared/src/shared/governed_exceptions.py`). Only `scripts/exception_register.py` loads the table. Without the declaration nothing checks that an owner is live or a row exists. The table's schema does accept such rows: any kebab-case id, four keys.
9. **`GitOps.reclaim_leaked` has no named caller or moment.** Draft 6 said "when the orchestrator starts"; draft 7 dropped it.
10. **Evidence index.**
    - `REPORT.md` (16:46) predates the draft: no rows for 89 of 105, 10 of 10, 12 s and 43 s, the 0.15 share, or `rule7`.
    - `scripts/sim/` is not staged.
    - Section 9's "9 to 17% less" was computed at u 0.07; at 0.15 it is 10% less to 3% more.
11. **Root testing is not measured.** 121 of 190 batches are stacked. A check that unlanded work added or changed is absent or different on member trees, so it ends unexplained. The 0.15 start comes from base-relative data. Also, the streak L1 does not count scans that attribute but send every member to the fallback.
12. **Small.**
    - Coalescing and `chain_snapshot` read `_chain_reservations.holds` directly, not `is_reserved`: three sites to compose.
    - `verdict_facts`' home is unstated, and σ3 no longer touches `observation.py`.
    - D9 says a file that does not collect "cannot be scanned", but 6.2 gives it no blocker. It is scanned and ends unresolved: the same result, three trees spent.

## 3. Leo's call

1. **What `q` means (A1).** Per exposure, counting scan-cleared requests as sound; or per request, counting every landing.
2. **D13's 5 to 8% (B1).** Keep it, or keep cleared members cleared in a fallback.
3. **Second-kind sizing (B2).** Size it directly now, or leave it to η with the figure recorded.
4. **A bound on releases (B4).**
5. **The 0.4 threshold against the 600 s budget (B5).**

## 4. Checked and sound

- **`rule7.py`.**
  - B1 (4 of 246, at thresholds), B2, every B3 figure, and D14's and D15's thresholds reproduce.
  - The solver converges in all 4,608 cells tried: q from 1e-9 to 0.999999, r to 0.9, u 0 to 1, caps 1/1 to 48/96, four starting points. At most 162 iterations; no dependence on the start.
- **6.1.** Bases that are an owner's own commit now resolve. An unresolved root only sends a red to the fallback.
- **Root plus member.** In 69 batches built on main, none of 270 members carried a main commit newer than the root (`staleroot2.py`). A merge that fails on the root is `MERGE_FAILED`, which ends released or fallback.
- **Attribution without `ALONE`.** I walked these cases: flaky beside a culprit, flaky alone, error on a member, error on the root, a failed merge, incomplete, main's with a culprit, ahead's with a culprit, and fails-only-together. Each ends culprit, cleared, released or one red set; the only unbounded path is B4.
- **A file that does not collect.** Its id is file-level, and `verify.py::_node_id_statuses` matches no testcase, so it is `ERROR` on tip and root: unresolved, fallback. The member is blocked by its own verify, not lost. `ABSENT` for a missing test needs one collect-only run, which σ1 names.
- **A 30% flaky test.** About 1% chance of a false conviction per batch, against 9% per solo verify today (hand calculation).
- **`note_blocked`'s trigger.** Voided, cancelled and abandoned requests never reach a delivered red.
- **`apply_scan`.** One row, so a re-run after restart applies once.
- **Decomposition.** σ3 and β2 share no file. ι1's sites match `PINNED_SITES`.
- **Section 2.** All five columns, the 75-minute line, the waits, L8's 7 to 14% and the break-even match sim-6. The first and third "differs from the simulation" statements hold; the second does not (B1).

## 5. Not checked

- No scan was executed on a real tree; the collect-only run's cost is unmeasured.
- Stacked batches: member tips against the root, and how often a member fails to merge on the root.
- Trees for a gate-rebased head (a base equal to `gate.rebased_sha`).
- Config guards for the new section (`defaults.yaml` parity).
- The simulator was not rerun; D13, root testing and direct sizing are unsimulated.
- Task texts were not re-read.
