# Addendum to section B: today's cascade, accidental unstacking and task 10995

**Correction to my previous answer.** Its base-choice gains assumed a main-based entry survives a red entry ahead. Under today's cascade that is false, and the best configuration's gain over the accepted baseline falls from +30 to +36% to +7.5 to +17%. It still needs the pool rule for stale passes. Narrowing the cascade is therefore worth 19–23 of those points and is a lane change in its own right.

Counting rules are as before: capacity-bound, e=0.01, main-red and restarts on, W1 unless stated, 120 h window, n=60 seeds; Δ% is the paired difference in landings per verify-hour, ± 95% CI. Hosts are idle 0–2%, so landings/h moves the same within 0.3 points.

## B under both cascades

Δ% vs the accepted baseline (always stack). "Today" tears down every entry behind a failed one; "narrowed" tears down only entries built on it. CI ±0.5–3.0.

| policy | stale pass handled by | cascade | q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|---|---|
| B1, main if P(tip tree good) < 0.5 | un-chained re-verify | today | −0.4 | −1.1 | −6.8 | −3.6 |
| | | narrowed | −18.5 | −15.9 | −13.4 | −7.9 |
| | same batch again | today | +0.3 | 0.0 | −1.4 | +0.4 |
| | | narrowed | −3.3 | −2.5 | −0.5 | +3.4 |
| | pool (known-good riders) | today | +0.6 | 0.0 | +0.3 | +1.0 |
| | | narrowed | +3.1 | +2.7 | +7.5 | +10.1 |
| B1, threshold 0.7 | un-chained re-verify | today | −10.1 | −13.6 | −30.5 | −21.0 |
| | | narrowed | −23.5 | −21.5 | −32.3 | −18.6 |
| | same batch again | today | −3.2 | −4.5 | −11.6 | −6.6 |
| | | narrowed | −4.7 | −5.6 | −8.1 | +0.7 |
| | pool | today | −1.0 | −1.7 | −1.1 | +0.5 |
| | | narrowed | +3.0 | +4.5 | +10.4 | +14.7 |
| B2, suspects only, always on main | re-verify (either) | today | −13.4 | −10.3 | −4.7 | −3.9 |
| | | narrowed | −11.3 | −8.9 | −3.9 | −2.3 |
| | pool | today | −4.1 | −1.3 | +0.1 | +1.1 |
| | | narrowed | −1.5 | +0.1 | +1.2 | +3.2 |
| Best configuration (threshold 0.7, sub-batch, closed-form depth, riders free) | un-chained re-verify | today | −45.6 | −39.3 | −27.7 | −20.5 |
| | | narrowed | −43.9 | −36.4 | −26.2 | −19.2 |
| | same batch again | today | +9.4 | +2.2 | −2.3 | −4.1 |
| | | narrowed | +17.6 | +9.5 | +6.1 | +2.3 |
| | pool | today | +17.0 | +15.1 | +9.9 | +7.5 |
| | | narrowed | +36.3 | +32.6 | +30.1 | +30.3 |
| Same without base choice | any | either | +9.0 | +3.4 | −0.8 | +1.3 |

- **B2 at threshold 0.5** is within ±1.3 of zero in every cell, except narrowed with pool at q=0.20 (+2.5).
- **Today's cascade matches your measurement.** No main-based entry landed on its own verdict in any run (measured: 0 of 39).
- **Why the best configuration stays positive under today's cascade with pool.** A torn-down entry that had already passed still leaves its members known-good, so main-based verifies act as pre-screens. At q=0.14 per run:

  | fate of the 250 entries built on main | count |
  |---|---|
  | landed on own verdict | 0 |
  | torn down by a cascade | 171 |
  | went stale | 75 |

  The first-run pass is the only thing that survives the tear-down.
- **Under the narrowed cascade with same-batch re-verify,** 17% of main-based entries landed on their own verdict (41 of 242 per run at q=0.14, with 84 stale and 49 re-verifies).
- **Evidence quality.** Red observations with unlanded entries in the tree are 2–3% for the best configuration under today's cascade with pool, against 64–77% for always-stack.

**Against your measured prior** (front lands on its own green 54%, red-then-blocked 22%):

| model, TODAY | front lands | front blocked | red batch requeued | main-red bounce | re-verify |
|---|---|---|---|---|---|
| always stacked | 61–70% | 10–20% | 12–17% | 5–7% | — |
| with accidental unstacking on | 52–60% | 8–19% | 14–17% | 6–7% | 8–12% |

With re-verify costing one verify and the front landing more often than it fails, building on main cannot pay on that prior. That is what the same-batch rows show. It pays only when a stale pass costs nothing.

## Accidental unstacking in TODAY

The toggle builds a share of dispatches behind a live entry on main unasked, under today's cascade with one full re-verify of a stale green. Δ% vs the same policy always stacked.

| calibration | what it reproduces | T, q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|---|
| share 0.265 | 28–29% of two-host overlap hours unstacked | −15.2 ±2.1 | −13.0 ±1.9 | −11.6 ±1.8 | −9.9 ±1.9 |
| share 0.15 | re-verify cost: 20–29 per 7.85 days, 4.0–5.6% of verify-hours | −9.4 ±2.0 | −7.4 ±1.8 | −7.7 ±1.6 | −5.4 ±1.7 |

- **The two calibrations disagree.** The overlap-calibrated share produces 33–49 re-verifies per 7.85 days and 6.6–9.8% of verify-hours, against your measured 23 and 4.4%. The cost-calibrated share reproduces the cost but only 16% of overlap hours. I report both as a bracket.
- **Value of an "always stack" rule for TODAY, as its own row:** +6 to +10% (cost-calibrated) up to +11 to +18% (overlap-calibrated).
- **W2** gives the same: −13.0 at q=0.08 and −12.0 at q=0.14 at share 0.265.
- **Other policies at share 0.265:** T+ loses 7–16% and the accepted baseline 3.5–12%. With un-chained re-verify the baseline loses 14–38%, because each stale batch becomes one solo per member.
- **The best configuration is unaffected** under narrowed cascade with pool; accidental main-based entries are just more riders.

Gains over TODAY-with-accidents (share 0.265), paired:

| design | q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|
| TODAY, always stacked | +17.9 ±2.5 | +14.9 ±2.2 | +13.1 ±2.0 | +11.0 ±2.1 |
| T+ (evidence fixes), always stacked | +30.2 | +29.0 | +25.0 | +17.9 |
| accepted baseline, accidents left in | +33.9 | +34.6 | +34.1 | +28.5 |
| accepted baseline, always stacked | +51.9 | +48.9 | +43.1 | +33.2 |
| best configuration (narrowed cascade, pool) | +107 ±5 | +98 ±4 | +86 ±4 | +73 ±3 |

## Task 10995

The toggle lets a green stacked verify land every in-flight entry its tree contains and cancel their verifies. Δ%, on vs off, same policy.

| policy | q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|
| TODAY, always stacked | +1.3 ±1.8 | +2.6 ±2.0 | +0.4 ±1.7 | +2.2 ±1.4 |
| TODAY with accidents (0.265) | +5.1 ±2.4 | +5.7 ±2.3 | +3.9 ±2.1 | +1.7 ±1.9 |
| T+ | +5.9 ±1.3 | +4.4 ±1.1 | +3.8 ±1.3 | +1.8 ±1.3 |
| accepted baseline | +3.2 ±1.1 | +3.2 ±1.1 | +2.8 ±1.0 | +1.7 ±1.0 |
| best configuration | 0.0 ±1.2 | 0.0 ±0.7 | 0.0 ±0.5 | +0.1 ±0.1 |

- **Deeper and sub-batched stacked variants** gain +0.7 to +3.5.
- **Mechanism:** 22–42 verifies are cancelled per 120 h run because a later stacked entry returned green first. Overrides of an undelivered red are 0.0–0.1 per run.
- **Does it change the ranking in A–C?** No.
  - **A:** the depth order under stacking is the same; deep sub-batching loses 0.5–2 points of its lead (q=0.05 table+2: +8.7 against +9.2).
  - **B:** the best configuration's lead over the baseline shrinks by about 4 points (+32.1 / +28.5 / +26.6 / +28.2), because the baseline gains about 3% and a mostly main-based lane gains nothing.
  - **C:** J-opt still ties static depth.

I did not add a wait-for-the-nearly-finished-verify variant.

## What this does to the recommendation

1. **Always stack** when no deliberate rule says otherwise. Fixing the accidental unstacking is worth +6 to +18% to TODAY on its own and needs nothing else.
2. **With neither lane change,** never build on main deliberately. Keep the batch-size rule from the previous answer (table at q̂ ≥ 0.10, table+2 with sub-batching below).
3. **With the pool rule but today's cascade,** the best configuration still earns +7.5 to +17% over the baseline, of which about half is base choice.
4. **With the pool rule and the narrowed cascade,** it earns +30 to +36%; this is the configuration my previous contract steps describe.
5. **With un-chained re-verify of a stale pass,** every deliberate main-based variant loses, whichever cascade.

## Simplifications that could change a ranking

- **[RANK] Accidental unstacking is an independent coin per dispatch.** The real causes cluster after failures (recovery after a dead tip, followers after a cascade, gate successors), and the two calibrations above disagree by a factor of 1.5–2.
- **[RANK] Known-good from torn-down passes.** Under today's cascade the policy still learns from a pass that returned before its entry was torn down. That is the whole of the positive result in point 3, and it rests on defects being deterministic and non-interacting.
- **[RANK] Model front-failure rate may be high.** With accidents on, the front fails 30–41% of the time against your 22% red-then-blocked plus an unknown part of the remaining 24%. The narrowed-cascade gain scales with that rate.
- **[RANK] 10995 scope.** Adoption applies only when the green entry is not stale; red batches already requeued at return are not recovered.
- **[RANK] Today's cascade also tears down every successor of a re-verified or dissolved head.**
- **[RANK] TODAY's stale re-verify** is the same members once more; the un-chained variant for TODAY releases links unmarked.

**Checks.** Defaults are unchanged from the engine before this addendum (64 of 64 runs identical). With e=0, nine toggle combinations over 144 runs landed no bad request and convicted or blocked no good one.

## Files

Everything is under `/tmp/gate-10069/sim/`; the repo is unchanged.

- **Engine:** `lane.py` (final), `lane_v2.py` (before this addendum)
- **Plans:** `plans2.py` (plans `XB`, `XB2`, `XT`, `X10995`)
- **Tables:** `results2_X_compact.txt`, `results2/{XB,XB2,XT,X10995}.txt`, built by `compactX.py`
- **Raw aggregates:** `results/{XB,XB2,XT,X10995}.json`
- **Validation:** `validate3.py`, `out_validate3.txt`
