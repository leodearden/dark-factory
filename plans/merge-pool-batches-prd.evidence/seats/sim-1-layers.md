# Gate 10069 modelling seat: merge batches chosen from failure evidence

**Headline.** Against TODAY, the evidence fixes plus persisted marks are worth +7 to +17% landings per verify-hour when main-red and restarts are on. Releasing a red batch's other members once its culprit is found, and blocking the last member by elimination, are worth another +11 to +18% (+8 to +12% without elimination at q ≤ 0.08). Three bookkeeping rules over red-batch member sets reproduce the joint-inference result within ±2%; the inference adds a principled on/off switch for elimination and little else measurable. The membership rule, sub-batches at table depth, uncapped batches, the void-charging objective and the per-request prior each earn about nothing or lose.

I found and fixed an engine bug mid-run: a parked request could be taken as a link and dispatched twice, which inflated J variants by 1–3%. Every figure below is from the fixed engine, with guards against double membership and double retirement. The earlier results are kept under `results_stale_doublemembership/` and are superseded.

**How figures are counted.**
- **Window and seeds:** 120 h window after a 24 h warm-up; seeds 0..n−1 shared by all policies in a cell. n = 100 at e = 0.01 in the main grids and rules plan, and in both sens_pol cells; n = 60 elsewhere.
- **Landings per verify-hour:** landings ÷ host-hours spent verifying, including cancelled part-runs, per seed; ±95% CI = 1.96·sd/√n. "Paired" means per-seed differences.
- **Void %:** verify-hours that were cancelled, returned then discarded, red only because of an entry ahead, or main-red.
- **Waits:** requests arriving in the first 60 h of the window, pooled over seeds; ">60" means the quantile falls among requests still waiting at the horizon.
- **Convictions right/wrong:** bad/good requests blocked without a verify of their own, per 100 landings.
- **Own-solo wrongful:** good requests blocked by their own solo false failure, per 100 landings.
- **"cap" cells:** capacity-bound; the waiting set is held at 75 (each retirement admits an arrival).
- **Policy defaults:** J uses e = 0.01 and θ = 0.9 unless stated.

**Policy names.**

| Name | Meaning |
|---|---|
| T | TODAY |
| T+ | fixes (i) and (ii), marks persist |
| TB | T+ with pool order and static table depth (a bridge to J's mechanics) |
| TR | TB plus three rules with no probabilities: unmark the rest of a red chain's members once one is blocked; test one member of a set at a time; block the last member when all others have landed |
| Jc / Jc1 / Jf / Jv | J-cap / cap+1 / J-free / J-void |
| J1 | suspects verified alone in pool order |
| J1d | J1 plus "one test per group at a time" (my addition) |
| J2 | sub-batch nearest 0.5, one test per group at a time |
| Otab / O12 | oracle at table depth / at depth 12 |

## 1. Validation

- **(a) Passes.** With a legacy single-stream RNG the new engine reproduces `engine_orig.py` run for run for I and M: 640 of 640 runs identical in landings, verifies and blocks (q 0.05–0.24, restarts off and on). `engine_orig.py` run here reproduces the published FINDINGS figures exactly.

  | q | I published | I new engine (n=200) | M published | M new engine (n=200) |
  |---|---|---|---|---|
  | 0.05 | 4.37 | 4.39 ±0.04 | 4.42 | 4.45 ±0.04 |
  | 0.08 | 3.74 | 3.79 ±0.06 | 3.75 | 3.86 ±0.06 |
  | 0.14 | 2.73 | 2.76 ±0.05 | 2.65 | 2.71 ±0.05 |
  | 0.24 | 1.89 | 1.90 ±0.03 | 1.78 | 1.81 ±0.03 |

  The new engine uses independent streams, so the small offset is seed-set noise: on disjoint seeds at n=600 the difference is −0.015 to +0.008 landings/h.
- **(b) Passes.** At q=0: T, T+, TB, Jc-J1, Jc-J1d, Jc-J2 and O12 land 12.00× OFF with 0.0833 verifies per landing; I 8.00×; M 6.00×; Jf (max 24) 24.00×. OFF is 2.653/h against 2/E[D] = 2.668.
- **(c) Passes.** With e=0 and no main-red: 3,840 runs, 24,272 convictions, 0 wrong, 0 good requests blocked, 0 bad landed. This covers Jc/Jf × J1/J1d/J2/J2-parallel with policy e=0, and policy e=0.01 at θ 0.8/0.9/0.95, restarts off and on.
- **(d) Passes.** Maximum |enumeration − brute force| is 3.3e-16 over four cases:
  1. Red {a,b,c}, prior 0.14, e=0.01: P(a bad) = q/[1−(1−q)³(1−e)] = 0.378067; P(a,b good) = 0.296795.
  2. Red {a,b} and {a,c}, priors 0.14/0.14/0.27, e=0: P(a bad) = qa/[qa+(1−qa)·qb·qc] = 0.811557; P(b bad) = 0.302061; P(c bad) = 0.407564.
  3. Red {a,b,c}, {c,d,e}, {e,f}, mixed priors, e=0.026, then a pass clearing d: all six marginals and one joint match.
  4. Nested {a,b,c,d} then {a,b}: P(a bad) = 0.516223 and P(c bad) = 0.083661 at e=0.01.
- **(e) For scale, model TODAY at W1.**
  - Verdicts returned per landing: 0.50 / 0.65 / 0.91 at q 0.05 / 0.08 / 0.14 without main-red, 0.61 / 0.80 / 1.14 with. Recorded: 0.50 and about 1.0.
  - Landings per verify-hour with everything on: 1.58 (q=0.08) and 1.10 (q=0.14) at W1; 0.86 and 0.61 at W2. Recorded: 1.3–1.5.
  - Landings per day: 53 (q=0.14, W1) and 41 (q=0.08, W2). Recorded: 40–48.
  - Red chains per 120 h at q=0.14: 58, of which 5.8 inherited-only and 9.0 main-red. Recorded: 58, of which 13 and 4.
  - So the model carries about half the measured inherited-red share and about twice the main-red share of red chains.

## 2. Main tables

**Capacity-bound, q=0.14, e=0.01, W1, main-red and restarts on (n=100).**

| policy | land/h | land per verify-h | vs T, paired | verifies/landing | void % | good wait p50/p95 h | bad→block p50/p95 h | solo/100 | convictions right/wrong | own-solo wrongful |
|---|---|---|---|---|---|---|---|---|---|---|
| T | 2.20 | 1.104 ±0.023 | — | 1.62 | 28 | 27/47 | 31/55 | 102 | 0/0 | 0.42 |
| T+ | 2.47 | 1.238 ±0.027 | +12% | 1.47 | 29 | 24/42 | 27/51 | 74 | 0/0 | 0.30 |
| TB | 2.48 | 1.247 ±0.027 | +13% | 1.46 | 29 | 25/39 | 26/40 | 73 | 0/0 | 0.26 |
| TR | 2.77 | 1.388 ±0.028 | +26% | 1.32 | 30 | 22/35 | 23/38 | 45 | 4.2/0.30 | 0.15 |
| Jc-J1 | 2.73 | 1.370 ±0.029 | +24% | 1.35 | 30 | 23/36 | 24/37 | 57 | 2.5/0.16 | 0.15 |
| Jc-J1d | 2.81 | 1.408 ±0.027 | +27% | 1.30 | 30 | 22/35 | 23/37 | 44 | 4.2/0.21 | 0.13 |
| Jc-J2 | 2.80 | 1.407 ±0.028 | +27% | 1.31 | 30 | 22/35 | 23/37 | 44 | 4.2/0.21 | 0.15 |
| Jc1-J1d | 2.78 | 1.396 ±0.031 | +26% | 1.34 | 30 | 22/36 | 24/38 | 60 | 2.2/0.11 | 0.21 |
| Jf-J2 | 2.55 | 1.281 ±0.038 | +16% | 1.51 | 33 | 24/41 | 26/43 | 69 | 2.7/0.11 | 0.22 |
| Jv-J2 | 2.72 | 1.363 ±0.036 | +23% | 1.41 | 32 | 22/38 | 24/41 | 52 | 4.3/0.13 | 0.13 |
| Otab | 4.15 | 2.081 ±0.024 | +88% | 0.86 | 28 | 14/26 | 14/26 | 24 | 0/0 | 0 |
| O12 | 7.24 | 3.637 ±0.054 | +229% | 0.55 | 42 | 8/18 | 9/19 | 32 | 0/0 | 0 |

**Same cell at q=0.08.**

| policy | land/h | land per verify-h | vs T, paired | verifies/landing | void % | good wait p50/p95 h | bad→block p50/p95 h | solo/100 | convictions right/wrong | own-solo wrongful |
|---|---|---|---|---|---|---|---|---|---|---|
| T | 3.15 | 1.581 ±0.040 | — | 1.11 | 26 | 19/41 | 23/57 | 58 | 0/0 | 0.23 |
| T+ | 3.54 | 1.776 ±0.039 | +12% | 0.99 | 26 | 18/34 | 21/43 | 50 | 0/0 | 0.24 |
| TB | 3.57 | 1.789 ±0.038 | +13% | 0.98 | 26 | 18/32 | 20/33 | 50 | 0/0 | 0.22 |
| TR | 4.07 | 2.041 ±0.038 | +29% | 0.87 | 27 | 16/29 | 17/31 | 31 | 1.6/0.20 | 0.10 |
| Jc-J1 | 3.96 | 1.987 ±0.039 | +26% | 0.90 | 27 | 16/30 | 17/31 | 39 | 0.1/0.00 | 0.11 |
| Jc-J1d (θ=0.9) | 3.93 | 1.973 ±0.040 | +25% | 0.91 | 28 | 16/29 | 18/31 | 36 | 0.2/0.01 | 0.12 |
| Jc-J1d (θ=0.8) | 4.07 | 2.041 ±0.040 | +29% | 0.87 | 27 | 16/28 | 17/31 | 33 | 1.5/0.19 | 0.08 |
| Jc-J2 | 3.91 | 1.961 ±0.041 | +24% | 0.93 | 30 | 16/29 | 18/31 | 23 | 2.1/0.02 | 0.09 |
| Jc1-J2 (θ=0.8) | 4.19 | 2.100 ±0.047 | +33% | 0.87 | 30 | 15/28 | 17/30 | 27 | 2.4/0.17 | 0.08 |
| Jf-J2 | 3.74 | 1.876 ±0.060 | +19% | 1.00 | 31 | 17/34 | 21/38 | 55 | 1.0/0.04 | 0.18 |
| Jv-J2 | 3.89 | 1.952 ±0.059 | +23% | 0.97 | 31 | 16/33 | 19/36 | 48 | 1.3/0.04 | 0.14 |
| Otab | 6.26 | 3.138 ±0.031 | +99% | 0.55 | 25 | 10/20 | 10/21 | 12 | 0/0 | 0 |
| O12 | 11.25 | 5.650 ±0.075 | +257% | 0.34 | 35 | 5/15 | 6/16 | 15 | 0/0 | 0 |

**Slow verifies (W2, median 75 min), everything on.** Landings/h, then landings per verify-hour.

| policy | q=0.14 land/h | q=0.14 per verify-h | q=0.08 land/h | q=0.08 per verify-h |
|---|---|---|---|---|
| T | 1.21 | 0.606 | 1.72 | 0.861 |
| T+ | 1.39 | 0.700 (+16%) | 2.00 | 1.004 (+17%) |
| TR | 1.57 | 0.787 (+30%) | 2.28 | 1.144 (+33%) |
| Jc-J1d | 1.59 | 0.798 (+32%) | 2.23 | 1.119 (+30%) |
| Jc-J1d (θ=0.8) | 1.59 | 0.798 | 2.30 | 1.153 (+34%) |
| Jf-J2 | 1.46 | 0.735 (+21%) | 2.13 | 1.067 (+24%) |
| Otab | 2.33 | 1.170 | 3.49 | 1.752 |

**"Today" open cell** (75 waiting, then arrivals 2.4/h, W2, everything on, n=60). Landings/h, waiting set at the horizon, good wait p50.

| policy | q=0.14 | q=0.08 |
|---|---|---|
| T | 1.20/h, 195 waiting, 79 h | 1.72/h, 139 waiting, 48 h |
| T+ | 1.39/h, 169, 62 h | 2.01/h, 101, 38 h |
| TR | 1.58/h, 143, 52 h | 2.27/h, 67, 28 h |
| Jc-J1d | 1.58/h, 142, 52 h | 2.22/h, 73, 31 h (θ=0.8: 2.29/h, 65, 29 h) |
| Otab | 2.32/h, 41, 26 h | 2.74/h, 6, 9 h |

At W1 and q=0.14 the same cell ends with 55 (T), 29 (T+) and 9 (Jc-J1d) waiting.

**Arrival-bound** (2/h, no backlog, W1, everything on). Every policy lands the same (1.70–1.72/h at q=0.14, 1.83–1.84/h at q=0.08). Only verifies per landing differ: 1.64 → 1.57 → 1.50 for T → T+ → Jc-J1d at q=0.14, and 1.34 → 1.28 → 1.26 at q=0.08. Waits are 1–2 h p50 and 9–15 h p95.

**Grid, landings per verify-hour** (capacity-bound, e=0.01).

| q | dur | env | T | T+ | TB | Jc-J1 | Jc-J1d | Jc-J2 | Jc1-J1d | Jf-J2 | Jv-J2 | Otab | O12 | T→T+ | T+→Jc-J1d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | W1 | off | 2.59 | 2.72 | 2.76 | 3.14 | 3.15 | 3.15 | 3.23 | 2.82 | 2.94 | 4.94 | 8.72 | +5% | +16% |
| 0.08 | W1 | off | 2.01 | 2.08 | 2.12 | 2.32 | 2.32 | 2.29 | 2.37 | 2.19 | 2.30 | 3.66 | 6.62 | +4% | +11% |
| 0.14 | W1 | off | 1.42 | 1.46 | 1.47 | 1.60 | 1.64 | 1.64 | 1.63 | 1.51 | 1.58 | 2.44 | 4.21 | +3% | +12% |
| 0.20 | W1 | off | 1.10 | 1.11 | 1.13 | 1.19 | 1.23 | 1.23 | 1.18 | 1.13 | 1.19 | 1.98 | 2.97 | +1% | +11% |
| 0.05 | W1 | on | 2.07 | 2.31 | 2.37 | 2.69 | 2.67 | 2.71 | 2.74 | 2.43 | 2.55 | 4.22 | 7.45 | +11% | +16% |
| 0.08 | W1 | on | 1.58 | 1.78 | 1.79 | 1.99 | 1.97 | 1.96 | 2.03 | 1.88 | 1.95 | 3.14 | 5.65 | +12% | +11% |
| 0.14 | W1 | on | 1.10 | 1.24 | 1.25 | 1.37 | 1.41 | 1.41 | 1.40 | 1.28 | 1.36 | 2.08 | 3.64 | +12% | +14% |
| 0.20 | W1 | on | 0.88 | 0.94 | 0.96 | 1.02 | 1.06 | 1.06 | 1.01 | 0.96 | 1.02 | 1.70 | 2.54 | +7% | +13% |
| 0.05 | W2 | on | 1.12 | 1.30 | 1.32 | 1.51 | 1.52 | 1.53 | 1.56 | 1.39 | 1.46 | 2.37 | 4.15 | +16% | +17% |
| 0.08 | W2 | on | 0.86 | 1.00 | 1.01 | 1.12 | 1.12 | 1.11 | 1.14 | 1.07 | 1.13 | 1.75 | 3.18 | +17% | +12% |
| 0.14 | W2 | on | 0.61 | 0.70 | 0.70 | 0.77 | 0.80 | 0.80 | 0.79 | 0.73 | 0.78 | 1.17 | 2.06 | +16% | +14% |
| 0.20 | W2 | on | 0.49 | 0.53 | 0.55 | 0.58 | 0.60 | 0.60 | 0.58 | 0.55 | 0.59 | 0.96 | 1.44 | +8% | +13% |

- **Effect of e (W1).** Moving e from 0 to 0.026 lowers every policy by 3–5% and leaves the order unchanged; the full rows are in `results/final_tables.txt`.
- **Wrongful blocks at q=0.14, main-red off.** Jc-J1d wrong convictions are 0 / 0.28 / 0.63 per 100 landings at e = 0 / 0.01 / 0.026. T's own-solo wrongful blocks are 0 / 0.34 / 1.0.
- **Total wrongly blocked good requests.** Elimination trades solo false blocks for wrong convictions about one for one. In the q=0.14, everything-on cell the totals per 100 landings are T 0.42, T+ 0.30, TR 0.45 and Jc-J1d 0.34.

**Does the membership rule bind?** Share of J clean batches where the rule left out a request the table would have taken:

| Variant | Share | What it is |
|---|---|---|
| Jc-J1d | 0.2–0.9% | effectively never binds |
| Jc-J1 | 7–13% | all provisional requests passed over |
| Jf-J2 | 5–18% | — |

Batches deeper than the table (cleared requests) are 0.8–1.2% of Jc-J1d batches; none are shallower.

**Batch sizes** (share of verifies started, q=0.14, everything on).

| Policy | Sizes |
|---|---|
| T | 1: 64%, 3: 32%, 4: 4% |
| T+ | 1: 51%, 3: 33%, 4: 16% |
| Jc-J1d | 1: 34%, 3: 65% |
| Jf-J2 | 1: 47%, 2–3: 11%, 7: 42% |

At q=0.08 Jc-J1d is 1: 40%, 4: 60%, and Jf-J2 batches are 12. At q=0.05 Jf-J2 batches are 19.

**Calibration** (Jc-J1d, q=0.14, no main-red; predicted P(pass) → observed).

| Entry position | Predicted → observed |
|---|---|
| Head of the frontier | 0.47 → 0.46 (n=1,337); 0.62 → 0.62 (n=9,541); 0.73 → 0.68 (n=75) |
| Stacked on unlanded entries | 0.02 → 0.04; 0.28 → 0.42; 0.36 → 0.49; 0.41 → 0.54; 0.63 → 0.64 |

The policy under-predicts stacked entries by about 0.13. I attribute this to a survivor effect: red verifies run shorter than green ones (35 vs 43 min), so an entry ahead that is still running is more likely green, and the model does not use elapsed time. I did not test that directly. With main-red on, the head bin 0.62 observes 0.53. All ten bins per policy are in the per-plan `.txt` files.

**Bounds, verifies per landing without lane mechanics.**

| q | Information bound | Two-stage at table depth | Oracle at table depth | Oracle at 12 |
|---|---|---|---|---|
| 0.05 | 0.30 | 0.45 | 0.25 | 0.14 |
| 0.08 | 0.44 | 0.58 | 0.34 | 0.17 |
| 0.14 | 0.68 | 0.81 | 0.50 | 0.25 |
| 0.20 | 0.90 | 1.03 | 0.58 | 0.33 |

The table's depth equals the two-stage optimum at every q. In the lane, Jc-J1d reaches 1.07 at q=0.14 without main-red; the gap to 0.81 is tear-down cost.

## 3. Gain decomposition

Paired, landings per verify-hour, everything on, e=0.01, W1, for q = 0.05 / 0.08 / 0.14 / 0.20; CI is ±0.6 to ±1.9 points.

1. **T → T+:** +11.5 / +12.3 / +12.1 / +7.2% (W2: +16 / +17 / +16 / +8%; without main-red and restarts: +4.9 / +3.9 / +2.6 / +1.0%).
2. **Its parts alone:** fix (i) +6.6 / +4.4 / +1.3 / 0; fix (ii) +3.3 / +1.3 / +0.4 / 0; persisted marks +0.7 / +2.4 / +4.8 / +3.8 (W2: +5 to +8.5). They interact and do not add.
3. **T+ → TB** (pool order, static depth): +2.6 / +0.7 / +0.7 / +1.7%. Mechanics account for about 1%.
4. **TB → unmark the rest once the culprit is blocked:** +12.6 / +9.7 / +4.2 / +1.3%.
5. **Add one test per group at a time:** −0.3 / −0.6 / −1.2 / −0.9%. Alone it earns nothing.
6. **Add block-the-last-by-elimination:** +2.7 / +4.6 / +8.1 / +8.8%. Without step 5 it earns about 0, because the last member is already in flight.
7. **Rules (TR) → joint inference (Jc-J1d, θ=0.8):** −0.9 / 0.0 / +1.4 / +1.4%. Equal within ±2%. At θ=0.9 it is −2.0 / −3.3 / +1.4 / +1.4%, because elimination is switched off at low q.
8. **J-cap+J1 over TB:** +13.9 / +11.1 / +9.9 / +6.4%. **J1 → J1d:** −0.8 / −0.7 / +2.7 / +4.0%.
9. **J2 sub-batches at table depth:** +1.2 / −0.6 / −0.1 / 0.0%. Nothing. J2-parallel loses 0–5% in the superseded pre-fix runs; it was not re-run on the fixed engine.
10. **J3 target P(pass) 0.5 / 0.65 / 0.8 with stacking:** 0 ±1%. **No-stack:** +11 to +16% per verify-hour, −14 to −19% landings/h, 25–28% host idle.
11. **Cap +1:** J1d +2.6 / +3.1 / −0.9 / −4.1%; with J2 and θ=0.8, +6.7 / +6.5 / −1.9 / −7.1%. It helps only at q ≤ 0.08.
12. **J-free (uncapped rule):** −9.0 / −4.9 / −9.0 / −9.1%. **J-void:** −4.5 / −1.0 / −3.2 / −3.9%, and unstable (CI up to ±39 points in regime cells).
13. **H, per-request prior** (12% at 0.27, rest 0.115; policy knowing the class vs lane-wide prior): +1.0 to +1.5% (±0.8 to ±1.4). An oracle at the same depth is +46%.
14. **Elimination needs the evidence fixes.** The same rules on T without them give 0.36–1.7 wrong convictions per 100 landings, against 0.12–0.30 with.
15. **Total T → Jc-J1d:** +29 / +25 / +27.5 / +21% (W2: +36 / +30 / +32 / +23%; without main-red and restarts: +21.5 / +15.5 / +15.4 / +12%).

## 4. What would change the ranking

- **Which resource counts.** No-stack variants win on landings per verify-hour (+11 to +27%) and lose on landings/h (−14 to −40%) with 25–41% idle. The stated figure of merit and the wall-clock rate disagree here; this is a call for Leo on whether idle verify hosts have value.
- **Swings in q.**

  | Swing | Jc-J1d over T+ (static prior) | With adaptive prior |
  |---|---|---|
  | 0.07 ↔ 0.21 every 24 h | +10 to +11% | about −1% vs static |
  | 0.05 ↔ 0.23 every 48 h | +5 to +7% | +8 to +10% over T+ |

  Under the larger swing J-free ties J-cap, and static-depth TB falls 3–4% below T+.
- **Prior error.** Within ±50% it costs at most 4%. A prior of 0.25 or more puts the table at depth 1 and J-cap falls 33% (below T).
- **The elimination switch.** It fires only when q/(q+(1−q)·e_policy) ≥ θ. That is 0.84 / 0.90 / 0.94 / 0.96 at q 0.05 / 0.08 / 0.14 / 0.20 with e=0.01, so θ=0.9 turns it off at q ≤ 0.08. Turning it off costs 0.2–0.9% at q ≤ 0.08 and 7–8% at q ≥ 0.14. At q=0.14, θ=0.95 or policy e=0.026 costs 6.7%.
- **Detection lag for main-red.** Throughput is unchanged; wrong convictions at q=0.14 go 0.17 / 0.21 / 0.30 per 100 landings for lag 0 / 1 / 2 verify durations.
- **Losing state at restart.** J loses 4–5%; T+ loses 5–9%.
- **Did not change the order:** e in 0–0.026, W1 vs W2, main-red or restarts alone, K = 2 or 4, waiting set 30, prompt tear-down of a red chain's followers, release ratio 1.05–1.5, deadline 2× / 4× / none, shedding superset observations. The aging pick lifts T by 4% and shrinks T → T+ to +7.5 to +8.4%.

## 5. Algorithm of the best variant (J-cap + J1d)

Persisted state: the cleared set; red observations (id → set of requests); the suspect clocks. Parameters: prior q per request, e, θ, release ratio 1.2, group cap 16, deadline = 2 × p95 verify duration.

1. **Tree of an entry.** At dispatch: its own members plus the members of every non-vacated entry ahead of it in the frontier.
2. **Observation, taken at verify return.**
   - If the verify was dispatched under main-red and returns after detection: nothing.
   - If it passes: every unlanded request of its tree becomes cleared and is removed from every red observation; an observation left empty is deleted.
   - If it is red and a tree member has since been blocked or convicted: nothing (explained).
   - Otherwise record the set of tree members that are unlanded and not cleared; if that set is empty, record nothing.
3. **Later edits.** A landing removes the request from every observation. A block or conviction deletes every observation containing the request. A solo red found to be main-red at the head deletes the observation it made. A cancelled verify makes no observation.
4. **Inference** runs after every verify return, applied verdict and restart, and before every pick.
   - Requests linked by shared observations form a group.
   - The posterior over the group's bad-sets is enumerated exactly: weight = product of priors × product over observations of (1 if the set hits it, else e). Each observation counts separately.
   - A group above 16 first sheds observations that are strict supersets of another; then its lowest-priority members leave the enumeration at their prior and are verified alone.
5. **Suspect.** A request is a suspect when its P(bad) exceeds 1.2 × its prior. This is recomputed each time; there is no sticky flag. Cleared requests have P(bad) = 0.
6. **Conviction check**, at every inference: any waiting request (not in flight, not reserved) with P(bad) ≥ θ is blocked with no verify. Its observations are deleted; repeat until none qualifies.
7. **Pick**, when a host is free and the frontier has a slot. First, the oldest waiting suspect that has been a suspect longer than the deadline is dispatched alone. Otherwise walk the waiting requests in pool (arrival) order:
   - The first non-suspect triggers a clean batch (step 8).
   - A suspect whose group has a test in flight is passed over. "In flight" means an unreturned entry holds a group member with raised P(bad).
   - Any other suspect is dispatched alone.
   - If nothing was dispatched and a passed-over suspect exists, the first one is dispatched alone. This keeps the lane work-conserving.
8. **Clean batch.** Let A be the unlanded members of the entries ahead; treat A as empty if P(A all good) ≈ 0. Walk waiting non-suspects in pool order. Skip a candidate c when 1 − P(A∪{c} good)/P(A good) > 1.2 × prior. Add c to S (n members) iff both hold:
   - (n+1)·P(A∪S∪{c} good) > n·P(A∪S good);
   - n+1 ≤ table depth at the per-item estimate (product of own P(good))^(1/(n+1)), counting cleared as 1.

   The first candidate is always taken. With one lane-wide prior the second condition is the table depth at 1−q.
9. **Placement.** The entry is appended at the frontier tail, stacked on every non-vacated entry ahead. Nothing in flight is displaced.
   - A red batch's members return to waiting at verify return, and its vacated entry keeps its slot until it reaches the head (today's mechanics).
   - A solo red is applied only at the head: block, or bounce if main-red.
   - A solo red behind a failed entry is discarded by the tear-down.
10. **Restart.** In-flight entries are torn down and their members return to waiting; all state from the first line survives.

J2 replaces "dispatched alone" in step 7 with the pool-order prefix of the group's waiting suspects whose P(tree passes) is nearest 0.5 (tie → shorter). It measured 0 ±1% at table depth.

The three rules (TR) reproduce steps 2–7 for single-culprit evidence:
- remember each confirmed red chain's member set;
- dissolve it when a member is blocked;
- test one member at a time;
- convict the last when the others have landed.

They must sit on fix (i).

## 6. Readings chosen, simplifications, and what I could not model

**Readings chosen where the brief was open.**
- P(pass S) is the chance the whole tree passes, including entries ahead.
- The "passed over given the entries ahead" skip in step 8 is my addition. Without it a batch re-takes requests whose status hinges on a verify in flight.
- J-cap's cap is applied per batch through the table (it equals the table depth at 1−q for a uniform prior).
- Group cap: shed supersets before falling back to solo.
- Solo main-red reds bounce rather than block in every policy, TODAY included.
- A void red under T+ is also kept out of the depth window.
- J1d and TR are variants I added.

**Contradiction.** J3 "nothing stacked until its verdict returns" cannot hold with "a free host with a pickable request always dispatches" once the entries ahead land. I implemented the hold and report the idle share; work conservation is relaxed for that sub-variant only.

**Simplifications.**
- **[RANK] Frontier mechanics come from the 8528 engine.** Followers of a red entry, and anything dispatched behind a vacated one, are torn down when it reaches the head. There is no re-verify when main moves (20–35% loss for all policies in that study), no adoption timing and no high lane.
- **[RANK] Defects are independent and deterministic.** There are no pairwise interactions and no bad request that sometimes passes. A pass clears every request in the tree, so a single false pass would be trusted.
- **[RANK] J knows q, and q is stationary** outside the regime and prior cells.
- **[RANK] Main-red.** Every verify dispatched in an episode fails; exactly two episodes (4 h and 10.5 h) per 120 h; no lane halt. The model's main-red share of red chains is about twice the measured one, so fix (ii) and persistence may be overstated.
- **[RANK] Inherited-only reds** are 10% of red chains against 22% measured, so fix (i) may be worth more than shown.
- **[RANK] Two dedicated hosts; idle time has no value.**
- **[RANK] Red verifies are shorter than green** (35 vs 43 min).
- A chain takes exactly one verify duration.
- False failures hit solo verifies as well.
- Restarts have zero downtime.
- Capacity cells use closed-loop arrivals, so waits there are essentially 75 ÷ throughput.

**Could not model.** Resubmission after a block and repeat offenders (a wrongful block costs one landing here, not human attention); footprints and conflicts; the real clique-minimal pick; the value of idle hosts to other projects.

## Files

Everything is in `/tmp/gate-10069/sim/`; nothing in the repo or `/tmp/8528-study/` was changed.

**Engine and runners**
- `lane.py` — engine and all policies
- `engine_orig.py` — untouched copy of the 8528 engine
- `gridlib.py`, `plans.py` — runner and plan definitions
- `report.py`, `summary.py`, `final_tables.py`, `bounds.py` — table builders
- `smoke.py`, `probe_cat.py`, `probe_k2.py`, `probe_kinds.py` — ad-hoc probes
- `grid.py`, `followup.py`, `validate.py` — copied 8528 helpers, unused

**Validation**
- `validate_a.py`, `validate_a2.py`, `validate_a3.py`, `validate_bcd.py`
- `out_validate_a.txt`, `out_validate_a2.txt`, `out_validate_a3.txt`, `out_validate_bcd.txt`, `out_bounds.txt`

**Results** (under `results/`)
- `final_tables.txt` — the tables quoted above
- `summary.txt`, `extras.txt`, `sens_compact.txt`, `sens_pol_j3_compact.txt`
- one `.json` and `.txt` per plan: `main_w1_off`, `main_w1_on`, `main_w2`, `open_arr`, `rules`, `sens_pol`, `sens_j3`, `sens_cell`, `sens_regime`
- run logs `out_<plan>.log` in the sim directory

**Superseded**
- `results_stale_doublemembership/` — pre-fix results and the staged pre-fix engine copy
