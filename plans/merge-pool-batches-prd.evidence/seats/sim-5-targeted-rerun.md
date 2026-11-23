# Fourth run: scan the failing tests to find the culprit (U+S)

**Verdict.** In this model the scan is worth far more than anything in the first three runs: at the default scan cost (5% of a verify per tree, linear) U+S lands +110% to +259% more per verify-hour than plain U at the 12/24 caps, with no wrongly blocked good request. Once reds are cheap the rule wants the largest batch it is allowed, so the caps and the scan cost set the result. The lane would stop being capacity-bound: it lands 7.6–16.9 requests/h against about 2.4/h arriving. The result rests on deterministic, non-interacting defects and a reproducible failing set.

**How figures are counted.** Capacity-bound (75 waiting), e=0.01, main-red and restarts on, always stacked, today's cascade, W1, 120 h window, n=60 seeds, same seeds per cell. Δ% is the paired difference in landings per verify-hour vs plain U (w=0, strict conviction), ± 95% CI. "Scan share" is scan time as a share of busy host time; it counts in the denominator.

## (e) Checks

- **Scan off:** every run reproduces the engine before this change (18 of 18).
- **e=0, no main-red:** no bad request landed, no good request blocked or convicted in four scan variants (12 runs each: linear A, binary B, miss 0.2 with standing flakes, free).
- **c_t→0, m=0:** a bad request is blocked 0.89 h after first dispatch on average, against a mean red verify of 0.61 h. That is about 1.5 verifies, not one. I attribute the gap to members that return at their prior when the base fails and to verifies torn down behind a failed entry; I did not decompose it.

## (a) U+S against U, the accepted baseline and TODAY

Default scan: c_t=0.05, linear, sound members return known-good, flake re-run turns a false failure into a pass, no misses.

| q | TODAY | baseline | U | U+S 12/24 | vs U | U+S 24/48 | vs U | U+S 48/96 | vs U |
|---|---|---|---|---|---|---|---|---|---|
| 0.03 | 2.73 | 3.53 | 4.02 | 8.44 | +110 ±4 | 9.66 | +140 ±5 | 9.83 | +144 ±5 |
| 0.05 | 2.06 | 2.65 | 2.85 | 6.59 | +131 ±6 | 7.73 | +171 ±6 | 8.14 | +185 ±5 |
| 0.08 | 1.57 | 2.04 | 2.12 | 5.24 | +147 ±5 | 6.67 | +214 ±4 | 7.38 | +248 ±4 |
| 0.14 | 1.11 | 1.40 | 1.40 | 4.24 | +204 ±4 | 5.78 | +314 ±4 | 6.80 | +387 ±4 |
| 0.20 | 0.87 | 1.05 | 1.06 | 3.80 | +259 ±4 | 5.30 | +401 ±4 | 6.36 | +501 ±6 |

Landings/h are twice these figures (hosts never idle). The 24/48 and 48/96 columns are the variant with the explicit rider batch; at 12/24 the two are identical.

Other measures, U → U+S at 12/24:

| q | verifies/landing | void % | scan share | batches red | mean batch | good wait p50/p95 h | bad→block p50/p95 h | wrongly blocked/100 |
|---|---|---|---|---|---|---|---|---|
| 0.03 | 0.48 → 0.16 | 32 → 34 | 15.8% | 61 → 44% | 9.7 → 14.5 | 7/24 → 4/12 | 14/33 → 4/8 | 0.04 → 0.00 |
| 0.05 | 0.69 → 0.21 | 32 → 39 | 20.6% | 64 → 57% | 8.0 → 14.9 | 11/27 → 5/14 | 17/34 → 4/13 | 0.08 → 0.00 |
| 0.08 | 0.88 → 0.26 | 31 → 39 | 21.0% | 56 → 62% | 4.7 → 13.7 | 15/30 → 6/16 | 19/34 → 6/15 | 0.24 → 0.00 |
| 0.14 | 1.32 → 0.34 | 30 → 34 | 18.8% | 53 → 59% | 2.5 → 11.9 | 23/36 → 7/17 | 24/39 → 6/16 | 0.40 → 0.00 |
| 0.20 | 1.80 → 0.39 | 32 → 31 | 18.4% | 60 → 58% | 2.3 → 11.2 | 28/41 → 7/18 | 30/43 → 6/17 | 0.38 → 0.00 |

- **Batch sizes at 12/24.** 12 members in 64–77% of batches. At q ≤ 0.08 a second mode of 17–23 (riders plus fresh) takes 21–23%; at q ≥ 0.14 the rest are 9–11.
- **Wrongly blocked.** Zero from the scan itself. With standing flakes it is 0.00–0.01 per 100; I did not trace its source.
- **Task 10995 on.** U+S vs U is +114 / +132 / +143 / +197 / +255. The ranking is unchanged.
- **Reference lane** (narrowed cascade, riders, base by cost): +92 / +144 / +181 / +219 / +262 vs U-with-base. The absolute level is higher at q ≥ 0.05:

  | q | U+S, phase-1 lane | U+S, reference lane |
  |---|---|---|
  | 0.14 | 4.24 | 5.23 |
  | 0.08 | 5.24 | 6.85 |

## (b) Sensitivity

Δ% vs the default scan config.

| change | q=0.05 | q=0.14 |
|---|---|---|
| c_t = 0.02 | +28 ±1 | +28 ±1 |
| c_t = 0.10 | −26 ±2 | −27 ±1 |
| c_t = 0.20 | −49 ±2 | −54 ±1 |
| binary search over prefixes | +32 ±1 | +25 ±1 |
| scan on a separate resource | +53 ±2 | +69 ±2 |
| sound members return at their prior (B) | −2 ±1 | −24 ±1 |
| miss rate 0.05 | −1 ±1 | −4 ±1 |
| miss rate 0.20 | −7 ±1 | −13 ±1 |
| false failure stands as a no-culprit red | 0 ±0 | 0 ±0 |
| conviction off everywhere | 0 | 0 |

- **Scan cost is the first-order input.** Even at c_t=0.20 U+S is still +18% (q=0.05) and +40% (q=0.14) over plain U (compounding the row above with the default gain).
- **Variant B matters only at high q.**
- **Conviction** never fired in the default configuration, so turning it off changes nothing.

## (c) What limits throughput once reds are cheap

- **The caps.** With its own cost the rule picks 12 at cap 12 and 24 at cap 24 at every q; at cap 48 it picks 40 at q=0.03 and 48 elsewhere. Lowering the fresh cap to 6 halves the gain (+42 to +144).
- **Scan time.** It grows with batch size: 16–21% of host time at 12/24, 21–26% at 24/48, 30–34% at 48/96. With a linear scan this is what bends the curve.
- **Tear-down behind a red batch.** Lost verify time is 18–21% at 12/24, 25–28% at 24/48 and 26–30% at 48/96 (U: 18–21%). It rises with batch size but is not the binding limit.
- **Riders waiting for a pass.** Known-good requests wait 1.0–1.5 h on average to land; 2–6% wait over 4 h and at most 1% over 12 h. Nothing starves.
- **Landing riders promptly.** The cost rule already does it. With about ten riders waiting, adding a fresh request costs more than its going cost, so the rule dispatches a riders-only batch. An explicit "batch of riders once 12 have accumulated" gives identical results at 12/24 and +3% at 24/48 at q=0.03 only. I did not test a wait-time trigger.
- **Waiting-set size.** A set of 150 gives the same figures at 12/24 and slightly higher at 48/96 (q=0.14: 7.02 against 6.80). The model's pool refills without limit, so it cannot show the lane running dry. At 7.6–16.9 landings/h against about 2.4/h arriving, the real lane would be arrival-bound and the backlog would drain.

## (d) U+S as contract steps

1. **When the scan runs.** A verify returns red, the red is not a detected main-red, and the flake re-run has not turned it into a pass.
2. **On what.** Re-run the failing tests on the entry's base, then on base+m1, base+m1+m2, and so on (k+1 trees), on the host that ran the verify. Binary search uses ⌈log2 k⌉+1 trees.
3. **Base fails.** Blame nobody. The entry is vacated and its members return to the pool at their prior, with no ledger entry.
4. **Base passes.** Every defective member is blocked now as a verify failure. Every other member is recorded known-good (variant A) or simply released (variant B). The entry is vacated; what was stacked on it is torn down as today.
5. **Scan cannot re-run the failing set.** Record the red observation as plain U does, with strict conviction.
6. **Cost rule.** C(S) = 1 + P_red·(c_scan(S) + Σ_i (1 − p_i/P_red)·a_nd(i)), with c_scan(S) = c_t·(k+1) and a_nd = a(0) under A or a(p_i) under B. With the flake re-run, P_red = 1 − g(S), where g(S) is the posterior probability that every member is good given the base is sound.
7. **Going cost.** a(0) = 1/(K(1−e)). For p>0, a(p) = min over k of [1 + R_k·c_scan(k) + (k·R_k − k·p)·a(0)]/k under A, or (1 + R_k·c_scan(k))/(k(1 − R_k + p)) under B.
8. **Front, members and caps** are unchanged from U: add c iff C(S∪{c}) − C(S) < a(p_c).

**Parts of plain U that become unnecessary** when misses are rare:
- conviction by elimination;
- deferral (there are almost no observations to wait on);
- the nested follow-up term in a(p).

Exact inference is kept only for the miss fallback and for standing flakes.

**New quantities to fit from rows:** c_t and the miss rate. q and e are still needed.

## Simplifications that could change a ranking

- **[RANK] One scan finds every culprit.** Defects are deterministic and do not interact. An interaction defect would pass every prefix test of its partner-less trees, and this model has none.
- **[RANK] Known-good is exact.** Variant A treats members that passed the failing tests as fully verified; variant B is the cautious reading and costs 24% at q=0.14.
- **[RANK] Scan cost is linear in trees and independent of tree size.** The host is held for the scan.
- **[RANK] The flake re-run is free** in the default reading.
- **[RANK] The pool refills without limit,** so the arrival-bound regime is asserted from the rates, not simulated.
- **[RANK] Not modelled:** the optional scan into the entries ahead; chain build time for 24–48-member batches; merge conflicts.
- **[RANK] Cap variants beyond 12/24 were not run** for the sensitivities, task 10995 or the reference lane.

## Files

Everything is under `/tmp/gate-10069/sim/`; the repo is unchanged.

- **Engine:** `lane.py` (scan in `Engine._scan_done`, cost in `UPolicy._cost`, `scan_cost_table`), `lane_v4.py` (before this run)
- **Plan:** `plans2.py` (plan `S`)
- **Tables:** `results2_S_compact.txt`, `results2/S.txt`, built by `compactS.py`
- **Raw aggregates:** `results/S.json`
