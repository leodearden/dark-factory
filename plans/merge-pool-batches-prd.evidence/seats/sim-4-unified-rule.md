# Third run: rule U as one integrated policy

**Verdict.** With the tear-down weight at 0, U matches the best stacked separate-policy variant within the confidence interval at q ≥ 0.05 and is 4.6 points behind at q=0.03. U as specified fails the e=0 check: it convicts good requests. A one-line restriction on conviction fixes that at a cost under one point. Extended to the base decision, U reproduces the threshold finding without tuning but earns 5–12 points less than the separate variant under narrowed cascade with riders, and nothing under today's cascade.

**How figures are counted.** Capacity-bound (75 waiting), e=0.01, policy e 0.009, θ=0.9, main-red and restarts on, always stacked, today's cascade, W1, 120 h window, n=60 seeds, same seeds per cell. Δ% is the paired difference in landings per verify-hour, ± 95% CI. Hosts are idle under 2%, so landings/h moves the same within 0.3 points.

## Checks before reporting

- **e=0, no main-red: fails for U as specified.**

  | variant | wrong convictions |
  |---|---|
  | w=0 | 7 of 485 (32 runs) |
  | base-by-cost variants | 5 and 9 |
  | w=1 | 0 of 305 |

  No good request was blocked by its own solo and no bad request landed in any variant. Mixing a red-set member into a batch with fresh requests lets a second red push its posterior past θ when the culprit is elsewhere.
- **Fix:** convict only the sole request left in a red observation. That gives 0 wrong in 2,556 convictions over 192 runs.
- **q=0:** every batch is 12 un-cleared members (1,280 of 1,283; the other 3 are 24-member batches).
- **a(q) against the second run:** not equal.

  | q | a(q) as specified | with last-member credit | second-run closed form, sub-batch | one at a time | Dorfman optimum |
  |---|---|---|---|---|---|
  | 0.03 | 0.256 | 0.247 | 0.209 | 0.277 | 0.341 |
  | 0.05 | 0.359 | 0.343 | 0.310 | 0.363 | 0.433 |
  | 0.08 | 0.494 | 0.467 | 0.407 | 0.453 | 0.540 |
  | 0.14 | 0.698 | 0.648 | 0.592 | 0.610 | 0.703 |
  | 0.20 | 0.826 | 0.783 | 0.732 | 0.732 | 0.826 |

  a(q) sits 13–23% above the sub-batch closed form because it charges every member of a red batch its full going cost even after the culprit is found. It has no release on block.

## (a) U against the baseline, the best stacked variant and TODAY

| q | TODAY | baseline | best stacked | U w=0 | U w=0 vs baseline | vs best stacked | vs TODAY | U w=1 vs baseline |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 2.73 | 3.53 | 4.14 | 3.95 | +11.7 ±3.6 | −4.6 ±2.5 | +44.6 ±5.0 | +19.8 ±2.7 |
| 0.05 | 2.06 | 2.65 | 2.89 | 2.87 | +8.2 ±2.8 | −0.8 ±1.8 | +39.4 ±4.5 | +5.1 ±1.4 |
| 0.08 | 1.57 | 2.04 | 2.11 | 2.14 | +4.8 ±1.6 | +1.3 ±1.4 | +35.7 ±2.6 | +2.2 ±1.0 |
| 0.14 | 1.11 | 1.40 | 1.39 | 1.40 | −0.3 ±1.1 | +0.5 ±1.6 | +26.2 ±2.6 | +0.5 ±0.7 |
| 0.20 | 0.87 | 1.05 | 1.06 | 1.06 | +1.2 ±1.0 | 0.0 ±1.1 | +21.5 ±2.1 | −13.9 ±2.0 |

Other measures, baseline → U w=0:

| q | verifies/landing | void % | wrong convictions/100 | good wait p50/p95 h | bad→block p50/p95 h |
|---|---|---|---|---|---|
| 0.03 | 0.48 → 0.50 | 23 → 32 | 0.00 → 0.07 | 9/22 → 7/25 | 12/25 → 14/34 |
| 0.05 | 0.66 → 0.69 | 25 → 32 | 0.01 → 0.03 | 13/25 → 11/27 | 15/27 → 17/34 |
| 0.08 | 0.87 → 0.87 | 27 → 31 | 0.18 → 0.15 | 16/29 → 15/30 | 18/31 → 19/34 |
| 0.14 | 1.30 → 1.32 | 30 → 30 | 0.22 → 0.28 | 22/36 → 23/36 | 24/38 → 24/39 |
| 0.20 | 1.82 → 1.80 | 32 → 32 | 0.19 → 0.20 | 29/43 → 28/41 | 30/44 → 30/42 |

- **w=1 is not robust.** At q=0.20 the tear-down term is charged to batches but not to the going cost, so 82% of verifies are singles.
- **The deadline never binds.** At most 0.1% of deferred requests wait past 2×p95, so the "none" variant is identical.
- **Strict conviction** changes throughput by −0.7 to +2.1 points.

## (b) Losslessness

With every request at one prior and a(p′) forced to 1, U picks the table's depth at 18 of 20 estimates from 0.75 to 0.995 (policy e 0.009). At two exact row boundaries it picks one less:

| estimate | table | U |
|---|---|---|
| 0.979 | 8 | 7 |
| 0.984 | 9 | 8 |

With e=0 it is also one less at 0.876 and 0.934. One grid step inside each row they agree. So the table is U's special case up to threshold rounding.

With its own going cost U picks 12 / 12 / 7 / 3 / 3 at q = 0.03 / 0.05 / 0.08 / 0.14 / 0.20 (w=0).

## (c) What U does with a member of a red batch (w=0)

| q | alone | sub-batch with its set | mixed with fresh | commonest batch sizes | fresh cap binds | deferral fires on |
|---|---|---|---|---|---|---|
| 0.03 | 13% | 13% | 74% | 12 (74%), 1 (20%) | 74% of batches | 54% of dispatches |
| 0.05 | 16% | 36% | 48% | 12 (49%), 1 (24%) | 49% | 56% |
| 0.08 | 36% | 31% | 33% | 7 (50%), 1 (31%) | 0% | 52% |
| 0.14 | 78% | 14% | 8% | 3 (69%), 1 (28%) | 0% | 39% |
| 0.20 | 98% | 1% | 1% | 3 (64%), 1 (35%) | 0% | 40% |

- **Deferral** passes over 3–10 requests each time. Deferred requests wait 0.45–0.48 h on average; 3% wait over 1 h. The work-conserving fallback never fired.
- **Value of deferral** (w=1, with vs without): −1.7 / +1.0 / +2.7 / +3.1 / +0.6 points.
- **The 24-total cap** binds in at most 0.2% of batches.

## (d) Information the table cannot use

- **(i) Heterogeneous priors.** Knowing the class is worth +2.2 ±1.3 to U w=0, against the baseline's +1.0 ±0.9. U w=0 starts 1.5 below the baseline in that cell, so the net over the baseline is +0.7 ±1.4. For w=1 it is −1.0 ±0.8.
- **(ii) Named culprit as a probability shift only** (U w=1, gain over U without the name):

  | q | a=0.5 | a=0.8 | a=0.95 | wrong convictions/100 |
  |---|---|---|---|---|
  | 0.05 | +11.9 ±1.6 | +40.3 ±2.3 | +56.0 ±2.8 | 0.09 / 0.33 / 0.33 |
  | 0.14 | +1.1 ±0.8 | +14.3 ±1.5 | +33.4 ±1.7 | 0.42 / 1.21 / 1.01 |
  | 0.05, strict conviction | +10.5 | +27.3 | +40.8 | 0.05 / 0.04 / 0.08 |
  | 0.14, strict conviction | +0.5 | +2.3 | +15.0 | 0.29 / 0.34 / 0.38 |

  The behaviour is inferred from the run counts:
  - The named member is verified alone when it reaches the front.
  - The rest of its set are deferred while that verify is in flight (not released at once), and freed when it blocks.
  - Once the shifted probability reaches θ (a=0.95, or after the observation shrinks) the unchanged conviction rule blocks the named member with no verify. So "block at once" emerges as well as "confirm first", which is where most of the gain and the extra wrong convictions come from.
- **(iii) Regime swings, adaptive vs static prior.**

  | policy | 24 h swing | 48 h swing |
  |---|---|---|
  | U w=0 | −3.4 ±2.0 | +10.4 ±3.7 |
  | U w=1 | −7.3 | −5.5 |
  | baseline | −1.2 | +2.5 |

## (e) U extended to the base decision

Δ% vs baseline. "Separate" is the second run's main-if-P<0.7 variant.

| lane mechanics | policy | q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|---|
| today's cascade, re-verify same batch | separate | +9.4 | +2.2 | −2.3 | −4.1 |
| | U w=0, base by cost | +8.2 | +4.8 | −0.3 | +1.2 |
| today's cascade, riders | separate | +17.0 | +15.1 | +9.9 | +7.5 |
| | U w=0, base by cost | +8.2 | +4.8 | −0.3 | +1.2 |
| narrowed, re-verify same batch | separate | +17.6 | +9.5 | +6.1 | +2.3 |
| | U w=0, base by cost | +8.3 ±3.6 | +9.1 ±1.8 | +4.7 ±1.1 | +2.9 ±1.5 |
| narrowed, riders | separate | +36.3 | +32.6 | +30.1 | +30.3 |
| | U w=0, base by cost | +28.6 ±4.7 | +26.0 ±2.2 | +17.9 ±1.5 | +25.4 ±2.2 |
| | U w=1, base by cost | +15.1 | +10.6 | +12.0 | +8.4 |

Does the cost comparison reproduce the threshold? Yes, without tuning. Share built on main by P(base sound), U w=1:

| mechanics | below 0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | above 0.9 |
|---|---|---|---|---|---|---|---|
| narrowed, riders | 100% | 100% | 99–100% | 53–100% | 33–87% | 24–56% | 0% |
| narrowed, re-verify | 100% | 41–86% | 2–35% | 0–3% | 0–2% | 0% | 0% |
| today's cascade | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

That is a crossover near 0.7 with riders and near 0.45 with re-verify. It matches the earlier finding that 0.5 beat 0.7 under re-verify.

## (f) Weak points

1. **Wrong convictions from mixing.** Fixed by convicting only the sole request left in a red observation.
2. **w=1.** Loses 13.9% at q=0.20. Carrying the tear-down term into the going cost repairs that (−1.0 ±1.4) but then loses 6.4% at q=0.14. Use w=0.
3. **q=0.03.** U w=0 is 4.6 points behind closed-form depth with sub-batching. Raising the fresh cap to 24 does not help (+0.7, inside noise). I believe the cause is a(p) overpricing follow-up (no release on block) and therefore mixing 74% of red-set members into fresh batches, but I did not isolate it.
4. **Today's cascade with riders.** U never builds on main and misses +7.5 to +17%. The spec prices a torn-down entry as a total loss; in the lane a pass that returned before the tear-down still makes its members known-good. Crediting that is the principled correction; I did not test it.
5. **Narrowed with riders.** 5–12 points behind the separate variant. Cause not isolated.
6. **Elimination refinement is not clean.**
   - Setting a(p)=0 above θ made U use fresh requests as test fodder (I dropped it after a 4-seed probe).
   - The last-member credit alone is unstable: with w=0 it gives −13.1 ±12.8 at q=0.03, with verifies per landing of 25–84 at q ≤ 0.08; with w=1 it ranges from +22.7 to −6.9.
   - Keep the plain form.
7. **Adaptive prior** hurts under the faster swing (−3.4 for w=0) and for w=1 in both.

## (g) U as contract steps (the tested form: w=0, strict conviction)

**Fitted from rows:** q (lane prior, or per class), e (false-failure rate of a sound tree), verify-duration p95.
**Fixed:** θ=0.9, 12 un-cleared and 24 total members.
**From the ledger:** the red observations (sets of unlanded requests), the known-good set, and the trees of unreturned entries.

1. **Posterior.** Exact inference per evidence group gives P(any set good). For a candidate set X built on base A: g(X) = P(A∪X good)/P(A good), and p_i = 1 − g({i}).
2. **Going cost.** a(0) = 1/(12(1−e)). For p>0, a(p) = min over k=1..12 of [1 + R_k·k·a(p/R_k)]/k, with R_k = 1 − (1−p)^k(1−e) and the k=1 term equal to 1. Tabulate on a grid of 4,000 steps, from p=1 downward.
3. **Batch cost.** P_red = 1 − (1−e)·g(S). C(S) = 1 for a single. Otherwise C(S) = 1 + P_red·Σ a(p_i/P_red), using a(0) for known-good members.
4. **Deferral.** Pass over a waiting request whose evidence group contains a request in the tree of an unreturned entry, while something else is pickable and it has been deferred less than 2×p95. If nothing else is pickable, dispatch the first deferred request alone.
5. **Front.** The first non-deferred request in pool order.
6. **Members.** Walk the rest in pool order. Add c iff C(S∪{c}) − C(S) < a(p_c). Known-good requests always join. Stop at 12 un-cleared or 24 total.
7. **Conviction.** Block a waiting request with p ≥ θ only if it is the sole request left in some red observation. A single red at the head blocks.
8. **Base, only under a narrowed cascade.** Compose S for each option and take the lower of E − Σ a(p_i):
   - E_tip = P_A·C(S) + (1−P_A)·(1 + Σ a(p_i))
   - E_main = C₀(S) + (1−P_red,0)·L·c_stale
   - C₀ and P_red,0 are computed without conditioning on A.
   - L = 1 − Π over head entries of (1 − (1−e)·P(their members good)).
   - c_stale = 1 for a re-verified batch, or |S|·a(0) for riders.

## Where I had to choose

- A red single costs no follow-up (it is blocked).
- "Fresh" for the cap means not known-good, not p>0.
- The conditional marginal after a hypothetical red is p_i/P_red, exact from the current posterior; no second inference pass is needed.
- D = min(K−1−position, hosts−1).
- The deferral clock starts at the first deferral.
- The name shift is a likelihood ratio on the named member tied to that observation, set so a lone observation gives the realised right-share (0.65 / 0.84 / 0.95).
- Under today's cascade E_main = 1 + F·Σa(p_i) + (1−F)·[follow-up + (1−P_red)·c_stale], with F the chance an entry ahead fails.

## Simplifications that could change a ranking

- **[RANK] Deterministic, non-interacting defects.** Riders never fail a batch, and a pass clears every member.
- **[RANK] Section (d)(ii) uses the realised right-share as known.**
- **[RANK] Section (e) keeps the earlier assumption** that an entry on main survives under the narrowed cascade.
- **[RANK] U with base choice was not run with strict conviction** in the throughput cells; it shows 0.03–0.46 wrong convictions per 100 there.
- **[RANK] Section (a) was not run under accidental unstacking or task 10995.**

## Files

Everything is under `/tmp/gate-10069/sim/`; the repo is unchanged and earlier policies still reproduce run for run (208 of 208).

- **Engine:** `lane.py` (class `UPolicy`, function `going_cost`), `lane_v3.py` (before this run), `stage2/upolicy.py`
- **Plans:** `plans2.py` (plans `Ua`, `Ud`, `Ue`, `Uf`)
- **Checks:** `checksU.py`, `out_checksU.txt`
- **Tables:** `results2_Ua_compact.txt`, `results2_Ud_compact.txt`, `results2_Ue_compact.txt`, `results2_Uf_compact.txt`, `results2/{Ua,Ud,Ue,Uf}.txt`
- **Raw aggregates:** `results/{Ua,Ud,Ue,Uf}.json`
