**Verdict: not a clean pass. 1 A-defect (a one-term slip the D13 edit introduced), 4 B.** Everything else from pass 6 is answered, and the cross-references hold.

Read-only against main `9856ed15dc`. Scratch is `/tmp/gate-10069/critic7/` (`d78.diff`, `rule8.py`, `c1.py`, `c2.py`). No repository test was run.

## 1. Pass-6 items against DRAFT 8

| Pass 6 | Status | What remains |
|---|---|---|
| A1 (rates) | answered | — |
| B1 (D13) | answered | the new formula has a slip (A1 below); D13 is not carried everywhere (B2) |
| B2 (second-kind walk) | left to η, as stated | the figure quoted for it is stale (B3) |
| B3 (pending conviction) | answered | a changed tip never lapses (B1) |
| B4–B9, B11, B12 | answered | — |
| B10 | partly | section 9's figure is stale again (B3); `REPORT.md` is out of this pass |

- **Rates.** I walked a batch of 12 through every exit: convicted, cleared then landed as cleared, fallback, released, released twice. Each new request writes exactly one `defect_rate` row.
- **Residual bias.** A cleared request made new again writes a second row. A cleared culprit that cannot be scanned is counted under `defect_rate`, not `cleared_defect_rate`. Both effects are under a tenth of the rate.
- **Kept kind.** "`forget` keeps the kind until the callback has read it" is implementable. Row 46's figures check: 12.8/70 = 0.18 and 0.2/50 = 0.004.

## 2. New defects

### A. Answer-changing

**A1. A lone new member's solo verify is priced at zero (6.5 "`F(n)` … for `n ≥ 2`, and 0 below" with "`(1 − s) · (F(f) + c · A_new)`"; B3 "11 cleared and 1 new member at 0.14"; D14 "one from 0.14").**

- In a batch of one new and c cleared members that takes the fallback, the new member verifies alone. That is one verify, but `F(1)` is 0.
- The first new member added to a cleared batch therefore looks cheap, and the second carries the whole of `F(2)`. The walk stops at one.
- At q 0.14 with 11 cleared: the first new member's marginal cost is 0.109 as written and 0.144 with the verify counted, against `A_new` 0.181 (hand-checked).

  | q (scanning on, cleared front, 20 new waiting) | as written | lone member priced at 1 |
  |---|---|---|
  | 0.11 | (12, 11) | (12, 11) |
  | 0.14, 0.17 | (1, 11) | (12, 11) |
  | 0.20 | (1, 11) | (0, 11) |

- By the contract's own arithmetic, (1, 11) plus 11 new elsewhere costs 10.7% more than (12, 11) at q 0.14 (`c2.py`). A cleared front follows nearly every red batch, so this is the common dispatch at dark_factory's rate.
- The fix is local: the going costs do not change, and new-front plans do not change. With scanning off, the cleared-front plans at q 0.03 to 0.08 also change: (1, 11) becomes (4, 11), (2, 11) and (0, 11).
- *Requirement:* a single new member in a batch of two or more costs one verify in the fallback. Recompute B3, D14 and `rule8.py`.

### B. Minor

1. **A pending conviction whose request shows another tip hides it for ever (6.4).** The request "stays in scan until the conviction is applied or lapses", a request with another tip "is left as it is", and a conviction lapses only "when its request leaves the lane". A hidden request cannot leave. Say that a tip mismatch lapses the conviction, and have row 30 assert the request is picked again.
2. **D13 is not carried into four places.**
   - `apply_scan`'s `FALLBACK` bullet still says "the members with that verdict are alone".
   - D12 still says "every member".
   - V1 says a `NONE` verdict clears nothing, but it now un-clears.
   - Row 47 covers only the blocker case, not a scan that ends in the fallback.
3. **Section 9 and D14 quote a stale price for joint sizing.** "Between 10% less and 3% more" was draft 7's formula. Under `rule8.py` a joint batch of 12 new and 11 cleared costs 9 to 22% less than two batches (11% at q 0.14). A new front still refuses cleared members from q 0.11.
4. **Small.**
   - The kept kind should be dropped by the callback whatever the status. A request with no done-callback (a coalesce train) would leak its entry.
   - The prose call `verify_alone(new_members, red_set=…, reason=…)` omits `scanning`.
   - A cleared-only batch that is not scanned writes no `alone` row and no `merge_scan` row. Its `uncleared` rows need the observation id for `unscannable_share`'s numerator.

## 3. Leo's call

1. **D13 against L9's words.** Cleared members of a fallback batch now return as new without verifying alone, including after a timeout or infrastructure verdict. L9 says "each member verifies alone". Ratify or restrict.
2. **Joint sizing stays with η at its real price (B3):** 9 to 22% on the affected batches. With A1 fixed the cleared-front case reaches 12 new at q 0.14; the new-front case still does not.

## 4. Checked and sound

- **`rule8.py`.**
  - It matches 6.5 as written, and its staged copy is identical.
  - B1 (4 of 246, at thresholds), B2, every B3 figure and D14's and D15's figures reproduce.
  - The solver converges in all 4,224 cells tried (corners, six cap pairs, four starts; at most 162 iterations).
  - The first kind's size is the per-member minimum in all 1,278 of its own cells.
  - The price of D13 is +1.2 to +2.2% on `A_new`, as stated.
- **D13's claim.** A cleared culprit that cannot be scanned becomes new, and its next red batch either convicts it or sends it alone. It cannot ride past two rounds.
- **Cross-references.**
  - Rows 1 to 49 are each assigned to a leaf, and ζ cites 1 to 49.
  - No `CONTROL_FAILED_TWICE` remains.
  - The method names in 6.4's prose match the code block, apart from the `scanning` argument (B4).
  - The checkpoint lists release counts.
  - The row kinds include `uncleared`.
  - 6.7's leaf list needs no addition.
- **Sites.** Both pick sites and all three reservation-read sites exist as named (`_acquire_next_request`, `_look_ahead`; `merge_lane/worker.py` at the `MergeIntake` wiring, `chain_snapshot` and the coalescing filter).

## 5. Not checked

- `REPORT.md` and the staging tree (excluded by the brief).
- Nothing outside the diff was re-read, apart from the cross-references above.
- No scan was run on a real tree and the simulator was not rerun; D13's new form is unsimulated.
