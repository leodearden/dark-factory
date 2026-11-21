# Follow-up sweep: batch size, sub-batching and what an entry is built on

**Headline.** Under today's stacking, larger batches earn little: +9% at q=0.05, +4% at q=0.08, nothing at q ≥ 0.14. The large gain comes from three things together: building an entry on main when the tip is risky, returning a stale pass's members to the pool as known-good instead of re-verifying them alone, and sub-batching. That combination gives +30 to +36% landings per verify-hour over the accepted baseline and +56 to +76% over TODAY (W1). With today's un-chained re-verify of a stale head, building on main is a loss of 4–52%, so that part of the design is conditional on the stale-pass rule. The principled membership rule (J-opt) ties the best static depth; it does not beat it.

**How figures are counted.**
- **Runs:** capacity-bound (75 waiting), e=0.01, main-red and restarts on, 120 h window after 24 h warm-up, same seeds for all policies in a cell.
- **Seeds:** n=60 per cell; n=50 for section E; n=30/40 for the no-stacking reference.
- **Δ%:** paired per-seed difference in landings per verify-hour, ± 95% CI in points.
- **Landings/h:** hosts are idle 0–2% in every work-conserving cell, so landings/h moves within 0.3 points of landings per verify-hour unless I say otherwise.
- **"base":** the accepted Jc-J1d (table depth, one suspect at a time, 2×p95 suspect deadline, θ=0.9, policy e=0.009).
- **Deadline:** every other J variant below runs without the suspect deadline; that alone is +0.4 to +1.3% at table depth.

**Checks on the new mechanics.**
- With base choice off, the new engine matches the engine frozen before this follow-up run for run (208 of 208).
- With e=0 and no main-red, 12 new variants over 768 runs gave 0 wrong convictions, 0 good requests blocked and 0 bad requests landed.
- The closed-form depth cost matches a one-host, one-entry simulation within 1–4% (simulation higher).

## 1A. Depth (entries always stacked on the tip)

Δ% vs base; CI ±1.0–2.3 (W1), up to ±2.7 (W2). J1d = one suspect at a time; J2 = sub-batch nearest P(pass) 0.5.

| q | isolation | +0 | +1 | +2 | +3 | +4 | +6 | ×2 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | J1d | +0.6 | +2.4 | +2.9 | +1.5 | −0.1 | −4.9 | −3.0 |
| 0.05 | J2 | +1.4 | +5.3 | +9.2 | +9.5 | +9.9 | +9.9 | +8.9 |
| 0.08 | J1d | +0.9 | +2.3 | +1.0 | −0.7 | −3.1 | −10.6 | −3.3 |
| 0.08 | J2 | +0.3 | +4.2 | +3.9 | +4.0 | +3.8 | +0.5 | +4.2 |
| 0.14 | J1d | +0.4 | +1.0 | −1.8 | −4.7 | −8.4 | −8.2 | −4.8 |
| 0.14 | J2 | +0.5 | −1.4 | −1.0 | −4.1 | −6.8 | −6.8 | −3.6 |
| 0.20 | J1d | +1.3 | −2.4 | −5.0 | −5.1 | −5.1 | −5.1 | −5.1 |
| 0.20 | J2 | +1.3 | −5.0 | −6.5 | −6.6 | −6.6 | −6.6 | −6.6 |
| 0.08 (W2) | J1d | 0.0 | +2.8 | +2.1 | −0.2 | −3.6 | −11.8 | −4.1 |
| 0.08 (W2) | J2 | −0.3 | +5.5 | +4.9 | +5.6 | +4.7 | +2.4 | +4.5 |
| 0.14 (W2) | J1d | +0.4 | +1.8 | −1.0 | −4.5 | −9.1 | −9.1 | −4.4 |
| 0.14 (W2) | J2 | +1.4 | −0.9 | −0.5 | −2.9 | −7.0 | −7.0 | −3.0 |

- **Optimum per q:** table+2 to +6 with sub-batching at q=0.05; table+1 to +4 with sub-batching at q=0.08; the table at q ≥ 0.14.
- **Where sub-batching starts to win:** from table+1 at q ≤ 0.08; never at q ≥ 0.14 under stacking.
- **Target 0.65 (J2'):** never better than 0.5 beyond noise, and worse from +4 at q ≤ 0.08.
- **Rule ceiling:** the n·P(pass) rule itself stops at 7 members at q=0.14 and 4–5 at q=0.20, so the deeper columns repeat.
- **Deadline:** keeping the 2×p95 suspect deadline costs deep batches another 1–3 points.

**Analytic optimum without stacking** (verifies per landing, e=0.01; one batch verify plus its follow-up, release on block, elimination of the last where θ allows):

| q | table depth | one at a time: at table → optimum | sub-batch 0.5: at table → optimum | information bound |
|---|---|---|---|---|
| 0.05 | 5 | 0.394 → 0.383 at k=7 | 0.385 → 0.327 at k=16 | 0.301 |
| 0.08 | 4 | 0.504 → 0.494 at k=5 | 0.486 → 0.443 at k=9 | 0.437 |
| 0.14 | 3 | 0.718 → 0.711 at k=4 | 0.718 → 0.689 at k=5 | 0.679 |
| 0.20 | 3 | 0.917 at k=3 | 0.917 at k=3 | 0.902 |

Stacking (2 hosts, K=3, no main-red) adds +28–36% verifies per landing for one-at-a-time at q ≤ 0.08 and +47–64% at q ≥ 0.14. For sub-batching it adds +44–75%, rising with depth. Of the gap between the analytic optimum and the simulated optimum, 87–100% is stacking loss (entries cancelled or discarded behind a failed entry, and reds inherited from an entry ahead). The rest is the lane settling on a shallower depth.

## 1B. What an entry is built on

Table depth, Δ% vs always-stack (B0); CI ±0.6–2.4. A passed entry whose base went stale is handled three ways:
- **solo:** each member verified alone again. I treat this as today's lane, per your note on 10721. I found no reference to 10721 in the merge-lane code; `_dispatch_rebased_head` rebuilds the head and parks every successor.
- **chain:** the same batch is verified once more at the head.
- **pool:** no re-verify of its own; the members return to the pool as known-good and ride later batches.

| variant | re-verify | q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|---|
| B1 main if P(tip tree good) < 0.3 / 0.5 / 0.7 | solo | −8.7 / −18.5 / −23.5 | −5.5 / −15.9 / −21.5 | −3.9 / −13.4 / −32.3 | −2.0 / −7.9 / −18.6 |
| B1, same thresholds | chain | −1.1 / −3.3 / −4.7 | −0.8 / −2.5 / −5.6 | −0.4 / −0.5 / −8.1 | +1.4 / +3.4 / +0.7 |
| B1, same thresholds | pool | +1.9 / +3.1 / +3.0 | +1.2 / +2.7 / +4.5 | +1.8 / +7.5 / +10.4 | +3.2 / +10.1 / +14.7 |
| B2 suspects only, 0.5 / always | chain or solo | −0.7 / −11.3 | −0.5 / −8.9 | +0.9 / −3.9 | +1.0 / −2.3 |
| B2 suspects only, 0.5 / always | pool | 0.0 / −1.5 | −0.1 / +0.1 | +1.3 / +1.2 | +2.5 / +3.2 |
| B3 hold behind a suspect test | — | +8.4 | +9.5 | +12.3 | +12.4 |

- **B3:** landings/h fall 17–22% with 26–29% of host time idle; it is not work-conserving.
- **Re-verify count:** B1 at 0.5 with chain re-verifies 20–25 entries per 120 h run (55–114 entries built on main).
- **Evidence quality** (share of red observations whose tree held unlanded entries ahead):

  | variant | share |
  |---|---|
  | B0 | 64–77% |
  | B1 at 0.7, pool | 15–59% |
  | B3 | 39–46% |
  | best configuration | 4–10% |

- **Combined with the best depths from A:** pool with B1 at 0.7 adds +9.5 / +16.4 / +11.3 at q = 0.05 / 0.08 / 0.14; at q=0.20 it loses a little to threshold 0.5 (+16.0).
- **Chain at 0.5:** adds +0.7 to +5.4.
- **Solo at 0.5:** loses 10–30%.

## 1C. A principled membership rule (J-opt)

**Closed form used.** For fresh members m with G = P(all good | base good) and g = G^(1/m):

R = [1 + (1−(1−e)G)·D + (1−G)·V(m,g) + e·G·Vf] / [(1−e)·G·m + (1−G)·L(m,g) + e·G·Lf]

- V, L are the expected verifies and landings of the isolation that follows a red batch.
- Vf, Lf are the same walk after a false failure.
- D is the number of verifies stacked behind.
- A candidate joins iff R falls.
- Known-good requests join without counting.

**Under stacking**, Δ% vs base:

| variant | q=0.05 | q=0.08 | q=0.14 | q=0.20 |
|---|---|---|---|---|
| J-opt, one at a time (picks 7 / 5 / 4 / 3) | +2.8 ±1.6 | +2.7 ±1.0 | +0.2 ±1.0 | +1.6 ±0.8 |
| J-opt, sub-batch (picks 12 / 7 / 5 / 3) | +8.9 ±2.2 | +3.1 ±1.5 | −0.8 ±1.2 | +1.3 ±1.0 |
| best static depth from A | +9.2 ±1.5 | +3.9 ±1.4 | +1.0 ±1.1 | table |

- **Does it beat static depth?** No; it ties it. It acts as a better table, not a better rule.
- **Two alternative objectives were worse:** weighting by the base's risk, and marginal cost against the lane's going rate.
- **Tear-down weight D** (0, 1 or 2) never changed the depth chosen.
- **How it binds differently from the table:** only through depth. Under stacking, known-good riders appear in 1% of batches and weakly implicated candidates are skipped ≤ 0.02 times per batch.
- **Riders matter once entries are built on main.** Batches then carry 2.8–5.8 riders. Taking riders outside the depth count is worth 10–15 points (for example q=0.14: +20.3 → +30.4).

**Regime swings** (q̄=0.14), Δ% vs base:

| policy | 0.07↔0.21 every 24 h | 0.05↔0.23 every 48 h |
|---|---|---|
| J-opt, adaptive prior | −0.6 ±1.5 | +6.1 ±2.5 |
| table+1, adaptive prior | −0.8 ±1.8 | +6.1 ±2.2 |
| table+1, lane's rolling estimate | −1.4 ±1.4 | +5.1 ±2.3 |
| table+1, static prior | +0.2 ±1.4 | +4.7 ±1.8 |
| J-opt, static prior | +0.5 ±1.4 | +4.9 ±1.8 |

J-opt with an adaptive prior does not beat table+δ with a rolling estimate. Adapting adds 0–1.4 points over static and costs 1–6 points when q is stationary.

## 1D. Lane shape for the best configuration

"Best" = main if P(tip tree good) < 0.7, pool rule, sub-batch 0.5, J-opt depth with riders free, no fixed suspect deadline.

| cell | base (land per verify-h) | best | best vs base | best vs TODAY |
|---|---|---|---|---|
| 2 hosts, q=0.02 | 4.59 | 6.84 | +49.1 ±2.9 | +102 ±5 |
| 2 hosts, q=0.03 | 3.53 | 5.14 | +45.6 ±3.2 | +89 ±5 |
| 3 hosts K=4, q=0.05 | 2.31 | 3.37 | +45.8 ±2.7 | +95 ±4 |
| 3 hosts K=4, q=0.08 | 1.71 | 2.51 | +46.2 ±2.4 | +91 ±3 |
| 3 hosts K=4, q=0.14 | 1.14 | 1.68 | +47.3 ±2.3 | +81 ±3 |
| 3 hosts K=4, q=0.20 | 0.82 | 1.35 | +65.3 ±2.4 | +81 ±3 |

- **Third host:** it costs the baseline 13–22% per verify-hour (more stacking) and the best configuration 1–8%.
- **Low q under stacking:** depth alone gives +21% (table+4) to +28% (×2) with sub-batching at q=0.02, and +19% to +23% at q=0.03.

## 1E. Culprit named by the failing test (an idea Leo has set aside for now)

Run under stacking, one-at-a-time isolation, n=50. A name is given 80% of the time. It is the culprit with accuracy a when a real defect is in the tree; false failures and undetected main-red name a random member. Realised right-share is 64 / 84 / 95%. Δ% vs base at table depth → table+4.

| q | confirm-first a=0.5 | a=0.8 | a=0.95 | block at once a=0.5 | a=0.8 | a=0.95 |
|---|---|---|---|---|---|---|
| 0.05 | +2 → +14 | +10 → +30 | +14 → +38 | +16 → +42 | +35 → +72 | +41 → +85 |
| 0.08 | −2 → +7 | +4 → +22 | +7 → +31 | +13 → +33 | +31 → +68 | +38 → +82 |
| 0.14 | −8 → −2 | −3 → +11 | 0 → +18 | +2 → +17 | +26 → +59 | +35 → +75 |
| 0.20 | −8 → −7 | −3 → +2 | 0 → +6 | −2 → +1 | +32 → +43 | +44 → +62 |

- **Does it move the optimal depth?** Yes. Without naming, table+4 is +0.6 / −3.3 / −8.7 / −5.2. With confirm-first at a ≥ 0.8 it beats table depth by 14–25 points at q ≤ 0.14.
- **Wrong blocks under block-at-once**, per 100 landings at q = 0.05 / 0.08 / 0.14 / 0.20:

  | a | wrong blocks |
  |---|---|
  | 0.5 | 4 / 7 / 13–14 / 22–24 |
  | 0.8 | 1.2 / 2.1 / 3.6–3.8 / 5.7–6.2 |
  | 0.95 | 0.4 / 0.5 / 0.9 / 1.4–1.5 |

- **Confirm-first** adds no wrong blocks.

## 2. Best overall configuration

Build on main when P(tip's tree good) < 0.7 or the tip is stale; stale passes return to the pool as known-good; sub-batch nearest 0.5; fresh-member depth from the closed form with riders outside the count; no fixed 2×p95 suspect deadline.

| cell | TODAY | base | best | verifies/landing (base → best) | void % | wrong convictions/100 | good wait p50/p95 h (base → best) | vs base | vs TODAY |
|---|---|---|---|---|---|---|---|---|---|
| q=0.05 W1 | 2.06 | 2.65 | 3.61 | 0.66 → 0.44 | 25 → 33 | 0.01 → 0.05 | 13/25 → 6/29 | +36.3 ±3.0 | +75.5 ±4.2 |
| q=0.08 W1 | 1.57 | 2.04 | 2.70 | 0.87 → 0.59 | 27 → 32 | 0.18 → 0.22 | 16/29 → 9/32 | +32.6 ±2.2 | +71.8 ±3.1 |
| q=0.14 W1 | 1.11 | 1.40 | 1.82 | 1.30 → 0.87 | 30 → 31 | 0.22 → 0.31 | 22/36 → 14/39 | +30.1 ±1.8 | +64.7 ±3.1 |
| q=0.20 W1 | 0.87 | 1.05 | 1.36 | 1.82 → 1.15 | 32 → 32 | 0.19 → 0.30 | 29/43 → 21/37 | +30.3 ±2.0 | +56.3 ±2.5 |
| q=0.08 W2 | 0.86 | 1.16 | 1.59 | 0.91 → 0.59 | 29 → 34 | 0.12 → 0.21 | 30/46 → 17/52 | +37.2 ±2.9 | +84.1 ±5.2 |
| q=0.14 W2 | 0.60 | 0.80 | 1.05 | 1.35 → 0.89 | 31 → 33 | 0.25 → 0.33 | 42/57 → 28/62 | +32.4 ±2.6 | +74.0 ±4.6 |

- **Landings/h** are twice the per-verify-hour figures (idle 0%); the Δ% are the same within 0.3 points.
- **Wrongful blocks in total** (convictions plus own-solo false failures) are unchanged: 0.11 / 0.27 / 0.34 / 0.46 per 100 for base against 0.08 / 0.27 / 0.37 / 0.48.
- **What each ingredient is worth:**

  | change to the best configuration | vs base at q = 0.05 / 0.08 / 0.14 / 0.20 |
  |---|---|
  | as specified | +36.3 / +32.6 / +30.1 / +30.3 |
  | without base choice | +9.0 / +3.4 / −0.8 / +1.3 |
  | with the 2×p95 deadline | +16.0 / +17.5 / +18.1 / +24.1 |
  | threshold 0.5 instead of 0.7 | +23.8 / +17.6 / +24.4 / +13.9 |
  | one-at-a-time instead of sub-batch | +12.8 / +16.7 / +25.0 / +30.3 |

- **Static depth does as well.** Table+δ with riders free reaches +41.9 (δ=4) / +31.5 (δ=2) / +30.4 (δ=2) / +30.2 (δ=0). At q=0.05 that is 5.6 points above J-opt, which overshoots to 12.
- **If a stale pass must be re-verified as a batch (chain):** the best is J-opt with sub-batching and threshold 0.5, at +13.8 / +8.1 / +6.9 / +4.9.

## 3. When larger batches win and when they lose

- **They win when a red batch costs only its own follow-up.** That holds when q is low (q ≤ 0.08), when isolation is by sub-batch rather than one at a time, and when nothing is stacked on the batch.
- **They lose under stacking as q rises.** A deeper batch is red more often, and each red discards the entry stacked behind it and voids evidence taken on top of it. At q=0.14 a depth-5 batch passes 47% of the time, so a second entry stacked on it passes 22%.
- **One-at-a-time isolation caps useful depth near the table**, because follow-up length grows with depth.
- **Building on main removes the stacking penalty.** Depth 5 with sub-batching then becomes the optimum at q=0.14 instead of a 1–2% loss.

## 4. Rule for a contract

**If the lane adopts the pool rule for stale passes:**
1. **Known-good.** A request that was in any passing tree is known-good until it lands.
2. **Riders.** Known-good waiting requests join the next clean batch in pool order without counting against its depth.
3. **Fresh depth.** The number of fresh members is the closed-form depth above, capped at table depth + 4. That gives 9 / 7 / 5 / 3 at q̂ = 0.05 / 0.08 / 0.14 / 0.20.
4. **Base.** Let T be the newest live entry and B the unlanded requests in T's tested tree. Build on main iff T is stale or P(all of B good) < 0.7; otherwise build on T.
5. **Verdict on main.** A red batch is an observation on its own members only. A red single blocks at the head. A pass lands at the head iff nothing has landed since it was built. Otherwise the entry is dissolved and its members become known-good riders, with no verify of its own.
6. **Tear-down.** A failed or dissolved entry tears down only the entries built on it.
7. **Isolation.** Sub-batch nearest 0.5, one test per group at a time, no fixed 2×p95 suspect deadline.

**If stale passes keep today's un-chained re-verify:** always stack on the tip. Keep the table at q̂ ≥ 0.10. Below that, use table+2 with sub-batching (+9.2% at q=0.05, +3.9% at q=0.08).

## 5. Simplifications that could change a ranking

- **[RANK] Known-good is exact here.** Defects are deterministic and do not interact, and there are no false passes, so a rider never fails a batch. The measured rate was 1 pairwise interaction in 58 red chains. This is the main risk to the +30% figure; rider batches reach 24.
- **[RANK] An entry built on main survives the failure of entries ahead**, modelled per the `_chain_dead_link` reading you suggested. Today's gate re-verify parks every in-flight successor; I tear down only those built on the re-verified or dissolved entry.
- **[RANK] Stale means any landing since the entry was built that is not in its tree.** Chain and solo re-verify happen in place at the head and need a host. A stale red single still blocks. Pool dissolves at once.
- **[RANK] Building on main is free.** It has no build time and no merge conflicts with a newer main, and a main-based verify takes as long as a stacked one.
- **[RANK] Deadline off.** The sweep runs without the suspect deadline. Good-wait p95 is 3–4 h longer at q ≤ 0.14 under W1 and 5–6 h under W2, and bad→block p95 is 5–7 h longer at q ≤ 0.14. A longer deadline (4–6×p95) is untested.
- **[RANK] The cap in step 3 is composed from tested points.** It was not run as one rule. At q ≤ 0.03 only uncapped J-opt was run.
- **[RANK] Section E was run under stacking only**, not with base choice.
- **[RANK] Closed loop of 75 waiting.** Hosts never idle, so the two throughput measures cannot diverge except under B3.
- **[RANK] Large suspect groups.** A group above 16 requests made of one observation is solved in closed form; with several observations it sheds supersets as before.
- **Unchanged from the first run:** main-red episodes, restarts, e, and the frontier mechanics.

## Files

Everything is under `/tmp/gate-10069/sim/`; the repo is unchanged.

- **Engine and model:** `lane.py` (final engine), `lane_v1.py` (frozen before the follow-up), `analytic.py` (closed form), `stage2/` (the patch scripts)
- **Plans and reports:** `plans2.py`, `best.json`, `rep2.py`, `compactA.py`, `compactAns.py`, `compactB.py`, `compactAB.py`, `compactC.py`, `compactE.py`
- **Validation:** `validate_regress.py`, `validate2.py`, `out_validate_regress.txt`, `out_validate2.txt`, `out_analytic.txt`, `out_jopt_depths.txt`
- **Full tables per plan:** `results2/{A,A_dl,B_table,B_best,AB,W,W2,C,C_regime,D,E}.txt`
- **Compact tables:** `results2_*_compact.txt`
- **Raw aggregates:** `results/{A,A_ns,A_dl,B_table,B_best,AB,W,W2,C,C_regime,D,E}.json`
- **Run logs:** `out2_*.log`
