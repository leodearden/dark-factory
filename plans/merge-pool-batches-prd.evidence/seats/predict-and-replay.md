# Gate 10069, analysis seat: per-request failure prediction for a merge-lane batch chooser

**Bottom line: no, do not build the chooser.** Per-request signal is weak out of sample, and both the replay and the cost arithmetic put the gain within about ±2% of the current table. Elimination conviction was wrong in 7 of 9 cases where the truth is known.

All figures are dark-factory only, from `data/orchestrator/runs.db` opened `mode=ro`, as of 2026-11-19T20:24Z.

Deploy instants come from the `runs` table and git ancestry of each run's `code_sha`:

| Change | Deployed (run start) |
|---|---|
| Un-chained mark, task 10617 | 2026-11-17T20:49:02Z |
| Mark kept until verdict applied, task 10723 | 2026-11-18T18:34:38Z |
| Depth table, task 8528 | 2026-11-19T04:25:50Z (the brief said ~04:15Z) |

## How the schema differs from the brief, and what I did instead

- **Chain membership is recorded only from 2026-11-15T11:19Z**, not ~11-12. Of 198 chain verifies since 10-12, 97 carry `chain_request_ids`; the other 101 (mostly 11-11 to 11-15) carry none. All replay work in B and C uses the 97.
- **`merge_queued` carries no `request_id` before 11-15.** I mapped each verify to the task's next `merge_finalized.request_id`, and took enqueue time as the task's first `merge_queued` after its previous request finalized.
- **`first_enqueued_at` is a task-level age anchor, not the request's enqueue time.** Task 5879's request, queued 11-19, carries a 10-19 value. I do not use it.
- **`workflow_verify.base_sha` is not updated on rebase.** Staleness is computed with git instead: commits from `merge-base(task-verify tip, main at t)` to main at t, where main at t is the last lane landing before t.
- **Coalesced trains (10-12 to 11-01T08Z) log a union verify as `chain_items=1`.** I identified 105 such rows by `train_started`..`train_merged`/`train_derailed` windows and excluded them from solo verdicts. Members of merged trains count as batch passes.
- **Only passing task verifies are recorded** (3521 of 3521 `workflow_verify` rows have `passed=true`), so "task-verify failures before enqueue" is not a possible feature.
- **Head sha at verify is not recorded** (`merge_queued.head_sha` on 44 rows only), so "unchanged content" in C is bounded, not measured.
- **Lane and tier exist only from 11-15.** `task_kind` has no variation (1232 of 1237 rows are `normal`).

## A. Predictability

**Dataset.** One row per merge request with its first attributable verdict, verdict time in 2026-10-22..2026-11-19, n = 1237, 186 failures (15.0%).

| Label source | n | Rule |
|---|---|---|
| Solo pass | 651 | first `chain_items=1` verify of the request passed |
| Solo fail | 186 | solo verify failed and was applied (`merge_attempt verify_failed` then `merge_finalized blocked: Post-merge…`) |
| Chain pass | 173 | member of a passing chain with recorded membership |
| Landed via chain, membership unrecorded | 89 | `done`, no solo verify, `landed_via_chain=1` |
| Train pass | 138 | superseded by a train that merged |

Excluded:
- 41 failed solo verifies that were never applied (void, stacked on a failed predecessor, main-health red).
- Requests with no verdict: 137 blocked "Suffix item needs rebase", 58 conflict, 48 "worker shutting down", 41 still open.

"All" below means all 1237 rows; "solo" means the 837 rows with a solo verdict. Solo-only overstates failure rates in the chain era because batch passes are dropped. "All" understates the rate for first-time requests in the train era, because trains excluded recently blocked requests. The truth lies between the two columns.

**Failure by era (all rows).**

| Era | n | Fail | Rate |
|---|---|---|---|
| 10-22..11-01T08 (trains) | 412 | 46 | 11.2% |
| 11-01T08..11-11 (singles) | 357 | 79 | 22.1% |
| 11-11..11-19T04:26 (chains) | 434 | 60 | 13.8% |
| Post-8528 | 34 | 1 | 2.9% (thin) |

Daily rates swing from 0/35 (11-10) to 22/49 (11-03) and 19/65 (11-15). Day alone explains 11.2% of label variance in-sample (29 groups); the best per-request cells explain 2.6–4.2%.

**Failure rate by bucket, fails/n.**

| Feature | Bucket | All | Solo | Discriminates (≥2×, n≥30)? |
|---|---|---|---|---|
| Prior solo-failed requests, same task | 0 | 142/1057 = 0.13 | 142/676 = 0.21 | 2.1× all, 1.5× solo; not monotone |
| | 1 | 40/140 = 0.29 | 40/127 = 0.31 | |
| | 2+ | 4/40 = 0.10 | 4/34 = 0.12 | |
| Re-submission | first | 122/952 = 0.13 | 122/591 = 0.21 | 2.1× all, 1.4× solo |
| | after verify block | 42/157 = 0.27 | 42/138 = 0.30 | |
| | after conflict | 20/108 = 0.19 | 20/98 = 0.20 | |
| Files changed (task-verify scope) | <6 | 58/467 = 0.12 | 58/311 = 0.19 | yes, 2.0–2.5×, but the top bucket is 5% of rows |
| | 6–24 | 83/491 = 0.17 | 83/338 = 0.25 | |
| | ≥25 | 17/56 = 0.30 | 17/46 = 0.37 | |
| Declared files (task store) | <12 | 121/969 = 0.12 | 121/639 = 0.19 | 1.9× / 1.7×; stable in all three eras |
| | ≥12 | 62/256 = 0.24 | 62/189 = 0.33 | |
| Modules changed | <3 | 134/913 = 0.15 | 134/616 = 0.22 | no (1.6×) |
| | ≥3 | 24/101 = 0.24 | 24/79 = 0.30 | |
| Task-verify age at merge-verify start | <0.5 h | n/a | 12/86 = 0.14 | 2.8×, but n = 23 fails the n≥30 test |
| | 0.5–24 h | n/a | 162/710 = 0.23 | |
| | ≥24 h | n/a | 9/23 = 0.39 | |
| Commits behind main at enqueue | 0 / 1–4 / 5–14 / 15–39 / 40–99 / ≥100 | 0.11 / 0.27 / 0.14 / 0.12 / 0.20 / 0.20 | 0.19 / 0.38 / 0.21 / 0.18 / 0.27 / 0.23 | no; not monotone |
| Queue wait | <1 h / 1–8 h / ≥8 h | n/a | 0.14 / 0.26 / 0.19 | no; not monotone |
| Priority | high / medium / low | 0.19 / 0.15 / 0.13 | 0.25 / 0.22 / 0.22 | no |
| Tier (≥11-15 only) | high / medium / low | 8/44, 3/32, 12/103 | n too thin | no |
| Complexity | simple vs other | 9/104 = 0.09 vs 0.16 | 9/58 = 0.16 vs 0.23 | no (1.8× / 1.5×) |
| Previously in a red batch | no / yes | 0.14 / 0.20 (n = 188) | 0.21 / 0.27 | no (1.3–1.4×) |
| Red chains before this verdict | ≥2 | 9/40 = 0.23 | 9/19 = 0.47 | n too thin |
| Hour of verify start (UTC) | 08–12 vs 20–24 | 0.08 vs 0.20 | 0.12 vs 0.28 | yes, 2.3×, but it is a clock property, not a request property |
| Runner | local vs laptop | n/a | 51/333 = 0.15 vs 135/504 = 0.27 | no (1.75×) |
| Verifies in flight at start | 0 vs ≥2 | n/a | 0.26 vs 0.18 | no |
| Base relation (≥11-07) | main vs stacked | n/a | 0.19 vs 0.25 | no |
| Lane-wide rolling estimate (8528-style, my reconstruction) | ≥0.93 | 15/225 = 0.07 | 15/204 = 0.07 | yes, 5–7×, but it is lane state, which the table already reads |
| | <0.6 | 40/111 = 0.36 | 40/76 = 0.53 | |

- **Lane-state confound.** Stratified by lane-state bucket (Mantel-Haenszel risk ratio, solo rows), prior-fail drops to 1.05 and re-submission after a verify block to 1.13. Declared files ≥12 holds at 1.67 and task files ≥12 at 1.54.
- **Staleness is confounded.** The "0–4 commits behind at verify start" bucket shows 35/73 = 0.48 failure, but 26 of those 35 failures fall on two days (11-03 and 11-15) when main was not moving because the lane was red.
- **Failures cluster by test.** Of the 186 solo failures, at least 63 (34%) first-name a repo-wide ratchet or guard test. `scripts.tests.test_inline_suppression_ratchet` alone accounts for 35 and `test_merge_gate_history_leg` for 13. The other 89 block reasons name no test in the first 300 characters.

**Best simple model.** Tables of 1–3 binary features, shrunk towards the base rate; trained on the first two-thirds in time (n = 824, 15.8% fail), tested on the last third (from 11-12T07:07Z, n = 413, 13.6% fail).

| Model | AUC | Brier (constant = 0.1172) | Safest group | Riskiest group |
|---|---|---|---|---|
| Lane bucket only | 0.542 | 0.1471 (worse) | predicted 0.07 → 3/32 = 0.09 | predicted 0.43 → 20/146 = 0.14 |
| Files≥12 + simple + ≥3 task verifies (chosen on train data only) | 0.554 | 0.1167 | quintile 12/82 = 0.15 | 14/54 = 0.26 |
| Prior-fail + declared≥12 (best on the test set, so optimistic) | 0.657 | 0.1125 | 42/361 = 0.116 | 14/52 = 0.269 |
| Re-submission + declared≥12 | 0.638 | 0.1144 | 23/270 = 0.09 | 24/102 = 0.235 |
| Lane bucket + re-submission + declared≥12 | 0.585 | 0.1351 (worse) | — | — |

- **Separation is about 2.3× at best.** The riskiest ~13% of requests fail at 26–27%; the rest at 11–12%.
- **Calibration** is fine for the per-request tables (predicted 0.124 → observed 0.103, n = 311; predicted 0.240 → observed 0.235, n = 102). The lane-bucket table is badly off out of sample.
- **The signal vanishes among requests that actually get chained.** Of 191 distinct requests in recorded chains, 164 have a verdict and 11 failed (6.7%). The prior-fail + declared-files flag marked 12 of them, of which 1 failed; the 152 unflagged had 10 failures.

## B. Gain on replay

**Break-even rule.** Under today's rule (a red chain costs one extra solo verify per member), pulling a member out of a chain of size k pays only if its own failure probability exceeds 1/k. That is 33% at k=3, 25% at k=4, 17% at k=6. No predicted group reaches 33%; the best observed is 27%.

**Replay of recorded chain rounds.** Verifies needed to resolve each recorded round under each policy:
- Never chain: k.
- Table only (actual membership plus the mark): 1, plus k if red.
- Chooser: one solo per flagged member, plus one chain of the rest, plus one solo each if that chain is red.

Predictions come only from the model trained before 11-12T07:07Z. "Avoided" means the unflagged remainder would have gone green.

| Sample | Chains (green/red) | Never | Table only | Chooser, prior-fail + declared≥12 | Chooser, train-selected model | Oracle (flags true culprits) |
|---|---|---|---|---|---|---|
| Post-mark, all (≥11-17T20:49Z) | 43 (30/13), 187 member-rounds | 187 | 106 | 105–110; 1–2 red avoided, 5–6 shrunk, 5 pulled from green | 112; 0 avoided, 5 shrunk, 6 pulled from green | 87–97 |
| Post-mark, pre-8528 | 30 (21/9) | 139 | 78 | 75–80 | 83 | 59–69 |
| Post-8528 | 13 (9/4) | 48 | 28 | 30 | 29 | 28 (no red chain resolved yet) |
| "Fresh" chains, whole window (no member already in a recorded red chain) | 34 (26/8) | 153 | 72 | 70–75 | 80 | 63–68 |
| All 97 recorded chains (dominated by the 11-16/17 re-chain cycle; not independent) | 97 (39/58) | 441 | 363 | 366–371 | 373 | 236–343 |

- **Ranges** are optimistic (unknown member outcomes count as pass) to pessimistic (count as fail). The two chooser columns flag at predicted ≥0.20.
- **At the cost-optimal threshold (above 1/3) the chooser flags nobody** and is identical to table-only.
- **Net effect post-mark:** −1% to +6% verifies against table-only (105–112 vs 106). Landings are identical across policies by assumption, so the same percentages apply to verifies per landing.
- **Per passing member-round, post-mark:** never 1.03–1.20, table 0.58–0.68, chooser 0.58–0.72.

**Cost arithmetic on the held-out groups.** Verifies per request for a chain of size k with per-request failure q and a chance e that a chain is red with no culprit member: 1/k + 1 − (1−e)(1−q)^k.

| Scenario (pooled q = 0.136) | One k for all | Risky group solo | Best k per group |
|---|---|---|---|
| Observed groups (88% at 0.116, 12% at 0.269), e = 0 | 0.687 at k=3 | +0.1% | −1.3% |
| Same, e = 0.10 | 0.748 at k=4 | −1.2% | −1.5% |
| Hypothetical 4× separation (80% at 0.085, 20% at 0.34) | 0.688 | −5.0% | −7.1% |
| Hypothetical 10× separation | 0.689 | −16% | −22% |
| Perfect prediction | 0.688 | −38% | −70% |

**Actual lane, for scale** (all `merge_verify` rows including re-verifies, per `merge_finalized done`):

| Period | Verifies | Landings | Verifies per landing |
|---|---|---|---|
| Singles era, 11-01T08..11-11 | 554 | 269 | 2.06 |
| Chains pre-mark, 11-11..11-17T20:49 | 434 | 305 | 1.42 |
| Post-mark, pre-8528 | 63 | 63 | 1.00 |
| Post-8528 (0.7 days) | 17 | 34 | 0.50 |

The replay's "never = k" understates a real never-chain lane, which also pays gate re-verifies.

**What the range is most sensitive to:**
1. **Whether any identifiable group fails more often than 1/k.** None does today.
2. **Red chains with no culprit member.** Of the 15 red chains whose members all later got a verdict, 7 had every member pass afterwards. No membership rule can avoid those.
3. **Unknown outcomes.** 26 of the 63 post-mark red-chain member-rounds have no later verdict.

## C. Joint inference and elimination conviction

**The mark, as the rows show it.** Post-mark and pre-8528, 9 red chains had 48 member-rounds. Their first follow-up verify was:
- solo for 11 (7 pass, 3 fail, 1 void);
- another chain for 30;
- nothing for 7.

Marks appear to be an in-memory set (`worker.py`, `_next_dispatch_unchained`, initialised empty), and the lane restarted at 11-18T18:34Z and 11-19T04:25Z. I did not trace whether the set is re-seeded on startup. Post-8528 there are 4 red chains with 15 member-rounds: 14 still await any verify and 1 passed alone.

**Elimination opportunities** (all but one member cleared before the last member's own verdict; passing in a later chain counts as cleared):

| Period | Red chains | Last suspect truly failed (saves 1 verify) | Last suspect actually passed (wrong conviction) | Truth never recorded | No opportunity | Unresolved |
|---|---|---|---|---|---|---|
| Since mark deploy (pre-8528 part) | 9 | 0 | 1 (11-18T19:08, task 8745) | 2 (tasks 5715, 10137) | 2 | 4 |
| Since 8528 | 4 | 0 | 0 | 0 | 0 | 4 |
| Pre-mark cycle era, for the error rate only | 45 | 2 | 6 | 2 | 4 | 31 |

- **Verifies saved since the mark deployed: 0 confirmed, at most 2.**
- **Wrong convictions: 1 of 1 with known truth since the mark; 7 of 9 over the whole recorded window.**

**Why exact inference would not convict at 0.9.** The posterior for the last suspect is q / (q + (1−q)·e), where e is the chance a chain is red with no culprit member.
- Observed e: 7/46 = 0.15 to 19/58 = 0.33 over the recorded window; 1/31 = 0.03 to 9/39 = 0.23 post-mark.
- Observed q among chain members: 0.07 to 0.20.
- The posterior tops out at 0.89 (q = 0.20, e = 0.03) and is typically 0.33–0.62.
- Reaching 0.9 at q = 0.136 needs e ≤ 1.75%.

**False-failure rate of a solo verify** (applied solo failures, 10-22..11-19, n = 186, all pre-8528):
- **Lower bound, 29/186 = 15.6%.** In 15, the next request passed with the same recorded tip and no agent invocation between; in 14, a steward re-queued and the landed diff is line-identical to the failed content.
- **Upper bound, 85/186 = 45.7%.** Next request passed and the recorded task-verify tip was unchanged or unknown.
- **Per solo verify, that is 3.5%–10.2%** (of 837).
- 26 of the 186 have no later request.

## D. Verdict

1. **Membership by per-request probability: do not build.** Held-out AUC is 0.55–0.66, the best group fails at 27% against a 33% break-even at k=3, and replay lands at −1% to +6% verifies.
2. **Joint Bayesian inference: do not build.** With 15–33% of red chains having no culprit member, the posterior does not reach 0.9.
3. **Elimination conviction: do not build.** It saved 0 confirmed verifies since the mark and was wrong in 7 of 9 known cases.
4. **Recording a prediction plus a readout: optional, only if nearly free.** Its one use would be to detect a feature whose flagged group fails above 1/k; it cannot pay for itself today.
5. **Larger levers the rows point at, outside my brief and not sized as designs:**
   - Marks appear to be lost across restarts.
   - 35 of 124 post-mark green-chain member-rounds were repeat passes; two requests passed 11 green chains each.
   - Post-8528, all 4 red chains were stacked (4 of 7 stacked red, 0 of 6 main-based red; thin).
   - At least 34% of solo failures name a repo-wide ratchet or guard test that a cheap pre-screen could catch.
6. **Resting on pre-8528 rows only:** all of A (post-8528 has 34 labelled requests, 1 failure), the false-failure rate (186 of 186), every conviction case with known truth, and 30 of the 43 post-mark chains in B. Post-8528 has 13 chains, 4 red, none resolved.

## Assumptions and weaknesses

- **Intrinsic outcomes.** A member's outcome alone is taken to equal its recorded solo verdict or a pass in a later chain. A chain is taken to be red exactly when a member fails alone, or for an external cause that would hit any sub-chain of two or more. Landings are assumed identical across policies.
- **Replay rounds are not independent.** Requests recur across rounds; the "fresh" subset is the control for this.
- **The oracle row is a lower bound on the true ceiling.** It cannot avoid red chains whose culprits are among the unknowns.
- **The lane-wide estimate is my reconstruction** from `merge_verify` rows, not the lane's own state. Only 13 rows carry the real `chain_depth_reason`.
- **Model choice.** The prior-fail + declared-files table was picked on the test set, so its 0.657 AUC is optimistic; the train-selected table is the honest figure. Tables are tiny, so quintiles contain ties and I report whole groups.
- **Trial-level features** (task-verify age, queue wait, runner, hour, in-flight count) exist only for rows with a verify, so they are read on solo rows only.
- **Task-store fields** are today's values, not the values at enqueue.
- **Main at time t** is the last lane landing, so direct commits to main are missed and staleness is slightly undercounted.
- **The line-hash content comparison** proves "identical" but over-reports "differs". Task 4513 was re-queued 6 seconds after a failure and still showed as differing.
- **Train-pass rows** carry the supersede time, not the train-merge time (task 5565: 04:36 vs 11:46). This affects era assignment only.
- **Cancelled verifies** (210 `merge_verify_cancelled` rows) are not counted as verifies in the replay.

## Could not determine

- Membership of the 101 chain verifies before 2026-11-15T11:19Z, including which members were in red chains from 11-11 to 11-15.
- The solo outcome of 26 post-mark red-chain member-rounds (14 post-8528 still queued; others abandoned, conflicted or blocked for rebase).
- Why 7 fully resolved red chains had no failing member (stacked-base inheritance, flake, or pairwise interaction). I did not read verify logs.
- An exact solo false-failure rate; only the 15.6%–45.7% bracket.
- Whether the un-chained mark is re-seeded after a restart.
- Any post-8528 per-request failure rate or chain red rate with a usable n.
- Whether a content-based feature (for example, a diff that adds inline suppressions) would isolate a group above 1/k. Not tested.

## Sanity check against raw rows

I traced six requests by hand against raw events; all labels matched.

| Request | Task | Raw events | Dataset label |
|---|---|---|---|
| `mr-8de9f874` | 4513 | solo verify failed 10-23T07:26:30, `verify_failed/branch_bug`, blocked | solo fail |
| `mr-77883711` | 4513 | re-queued 6 s later, passed 11:16, gate re-verify passed, done | solo pass, 1 prior fail, re-submission after verify block |
| `mr-876d7523` | 10754 | member of the green k=3 chain at 11-19T18:25; enqueue 11-18T19:17 survives the 04:25 restart re-queue | chain pass |
| `mr-88f3f0ca` | 8826 | red k=6 chain 11-18T19:35, cancelled verify 01:19, solo fail 02:02, blocked 02:49 | solo fail, 14 red chains before |
| `mr-e2463a43` | 4650 | no verify under its task, `landed_via_chain=1` on 11-14T06:07 | landed via unrecorded chain (pass) |
| `mr-a7f0fb40` | 5565 | superseded by `coalesce-7712-3a9fa5e6`, which merged 10-30T11:46 | train pass |

The check exposed two bugs, both fixed before the figures above: the `first_enqueued_at` misuse (via `mr-a57b9dc0`, task 5879) and the stale `base_sha` staleness. It also exposed the train-pass timing inaccuracy noted under weaknesses.

## Scripts and outputs

Everything is in `/tmp/gate-10069/predict/`; nothing in the repo was touched.

| Step | Script | Output |
|---|---|---|
| Extract lane events since 10-12 | `extract.py` | `events.jsonl`, `inv.jsonl` |
| Requests, trials, labels | `build.py` | `requests.json`, `trials.json` |
| Features | `features.py` | `rows.json`, `rows_all.json` |
| Bucket tables (A) | `univariate.py` | `univariate.out`, `rows_window.json` |
| Holdout models (A) | `model.py` | `model.out` |
| Chain replay (B) | `replay.py` | `replay.out`, `chain_recs.json` |
| Cost arithmetic (B) | `analytic.py` | `analytic.out` |
| Elimination and external-red rate (C) | `elim.py` | `elim.out` |
| False-failure bounds (C) | `falsefail.py`, `falsefail2.py` | `falsefail.out`, `falsefail2.out` |
| Raw timeline for hand checks | `tl.py`, `chains.py` | — |
