# Gate 10069 measurement seat: merge-lane re-measure on recorded rows

The new numbers do not support "per-request failure probability" batching as a capacity gain. A chain verifies in the same time as a single request, so chains are almost free. A single request passes about 0.80 to 0.82 of the time, and the implied per-request rate inside chains is about the same. The 8528 table is already more cautious than the rule "admit the n-th member while q < 1/n", so in window (b) the rule never binds.

The real loss is after a red chain. Most members never got their solo verify before being chained again. Since 10617 deployed, every member that was re-chained before its solo had an orchestrator restart in between.

**Data used.** Store: `/home/leo/src/dark-factory/data/orchestrator/runs.db`, opened read-only with `mode=ro`. Merge events since 10-20 were cached to `/tmp/gate-10069/measure/events.json` by `extract.py`, and task activity events to `task_events.json`. `/tmp/gate-10069/measure/git_branches.json` holds the commits of each landed branch, taken from `git log` of the "Merge task/X" commits on main since 11-03.

**Windows.** Event timestamps are ISO strings ending `+00:00`. `first_enqueued_at` in queue events is an epoch float.
- **Window (a):** 2026-11-05T00:00Z up to 2026-11-19T04:25:45Z.
- **Window (b):** from 2026-11-19T04:25:45Z to the end of the data, about 20:24Z. It is a single orchestrator run, about 16 h long.
- **The (a)/(b) boundary is the 8528 deploy.** 8528 merged at 03:18Z. The orchestrator restarted between 04:25:35Z (last event of run-9a6d5c55b0a4) and 04:25:53Z (first event of run-c1a1cfef15e7). Every `merge_verify` row after the boundary carries `chain_depth_reason`, and no row before it does.
- **10617 deploy:** merged 11-17T18:45Z, live from run-7c108b742770 at 11-17T20:49Z. The first red chain after it was 11-17T21:42Z, on task 7736.
- **The table changed inside (b).** 10920 retuned the depth table on 11-19, merged 11:15Z. Rows show `row:[0.85,4]` up to 13:43Z and `row:[0.75,3]` from 15:26Z. So (b) contains two tables.

**Exclusions everywhere.** Runner `history-leg` is excluded (1 row). Durations are `data.duration_ms`.

**Gate re-verifies are identified exactly.** A gate re-verify is a `merge_verify` whose `merge_sha` equals the `merge_sha` of some `merge_reverify_gate` event. That finds 169 rows, all single requests, all in (a). The `runner_reason=gate_on_head_host` field catches only 86 of them.

## 1. Single-request pass rate; chain pass rate by size

Script: `/tmp/gate-10069/measure/items12.py`.

**Isolation solos.** An isolation solo is a task's first single verify after it was in a red chain. Chain members are only recorded from 11-15T11:19Z (`chain_task_ids`), so before that date isolation solos cannot be tagged.

| window | population | passed/n | rate | Wilson 95% |
|---|---|---|---|---|
| a | all singles | 545/637 | 0.856 | 0.826–0.881 |
| a | **singles without gate re-verifies** | **376/468** | **0.803** | 0.765–0.837 |
| a | singles without gate re-verifies or isolation solos | 357/434 | 0.823 | 0.784–0.856 |
| a | gate re-verifies only | 169/169 | 1.000 | 0.978–1 |
| a | isolation solos only | 19/34 | 0.559 | 0.395–0.711 |
| a | without gate re-verifies, laptop / local | 217/278 = 0.781 · 159/190 = 0.837 | | |
| b | singles without gate re-verifies | 4/4 (2 of them isolation solos) | 1.000 | 0.51–1 (**too thin to use**) |

Chains, gate re-verifies excluded (no chain was ever a gate re-verify):

| window | size | passed/n | rate | Wilson 95% |
|---|---|---|---|---|
| a | 2 | 13/19 | 0.684 | 0.46–0.85 |
| a | 3 | 22/57 | 0.386 | 0.27–0.52 |
| a | 4 | 2/5 | 0.400 | 0.12–0.77 |
| a | 5+ (13 of size 5, 85 of size 6) | 44/98 | 0.449 | 0.35–0.55 |
| a | all chains | 81/179 | 0.453 | 0.38–0.53 |
| b | 3 | 3/4 | 0.75 | 0.30–0.95 |
| b | 4 | 6/9 | 0.667 | 0.35–0.88 |
| b | all chains | 9/13 | 0.692 | 0.42–0.87 (**thin**) |

The daily single pass rate is not stable. It was 0.90–0.97 from 11-05 to 11-10, 0.67 on 11-12, **0.24 on 11-15** (6/25), 0.50 on 11-17 (6/12) and 0.94 on 11-18.

## 2. Implied per-request pass rate in chains (P^(1/n))

Same script as item 1. The single rate used for comparison is 0.823 (gate re-verifies and isolation solos excluded). "Expected passes" assumes each request in a chain fails independently at that single rate.

| window | size | implied per-request rate (CI carried from the Wilson bounds) | expected passes vs actual | z |
|---|---|---|---|---|
| a | 2 | 0.827 (0.68–0.92) | 12.9 vs 13 | +0.07 |
| a | 3 | **0.728** (0.65–0.80) | 31.7 vs 22 | **−2.59** |
| a | 4 | 0.795 (0.59–0.94) | 2.3 vs 2 | −0.26 |
| a | 5+ (mean size 5.87) | **0.872** (0.84–0.90) | 31.2 vs 44 | **+2.77** |
| a | all (mean size 4.49) | 0.838 (0.81–0.87) | 78.1 vs 81 | +0.45 |
| b | all (mean size 3.69) | 0.905 (0.79–0.96) | no single rate to compare (n=2) | – |

- **Pooled over all sizes, the chain rate matches the single rate** (0.838 vs 0.823, confidence intervals overlap).
- **By size it does not.** Size-3 chains fail more than independence predicts, and size-5+ chains fail less. Failures are therefore not independent across chain size.
- **A likely reason, not tested:** before 10617, many size-3 chains were halved re-chains of a red chain's members, so they carried known-suspect members.

## 3. False-failure rate of single-request verifies

Script: `/tmp/gate-10069/measure/item3.py`.

**Population.** Failed single verifies that were not gate re-verifies. For each, the "next verify" is the task's next `merge_verify`, either alone or as a chain member (membership is only visible from 11-15). **Window (b) has no such failures (n=0).**

| window (a), n=92 failed | count | share |
|---|---|---|
| next verify passed | 34 (7 of them as a chain member) | 0.370 |
| next verify failed | 40 | 0.435 |
| no later verify | 18 | 0.196 |
| R1: next passed, and no `task_started` or `phase_enter` in between | 20 | 0.217 |
| R2: R1, and no agent `invocation_end` in between (only `unblock_auto` allowed) | 5 | 0.054 |
| Git check: landed branches only (n=25), no commit authored between the two verifies | 8 | 0.32 |
| Git check and R2 both hold | 3 of 25 | 0.12 |

Weaknesses of the proxies:
- R1 lets a steward fix through, so it is an upper bound. R2 is close to a lower bound.
- The git check reads author dates, so an amended commit is invisible to it. It also sees landed branches only.
- No event records the head sha at verify time. `merge_queued.head_sha` is filled on 44 rows since 11-16 and on none of these pairs.

**Bound on false failure per verify of a sound tree.** A sound tree here means a pass or a false failure.

| basis | false failures / sound-tree verifies | rate | 95% upper bound (one-sided Clopper–Pearson) |
|---|---|---|---|
| R1, non-gate singles | 20/396 | 0.051 | 0.073 |
| R2, non-gate singles | 5/381 | 0.013 | 0.027 |
| R1, gate re-verifies counted as sound trees too | 20/565 | 0.035 | 0.051 |
| R2, same | 5/550 | 0.009 | 0.019 |
| Direct: gate re-verifies (an already-green tree, rebased onto a moved main) | 0/169 | 0 | **0.018** |

Two caveats:
- The lane suppresses known flakes (`merge_flake_suppressed`, 48 events in the window). These rates are what remains after suppression.
- Gate re-verifies completed only about once a day after 11-14. Most gate events after that were cancelled with cause `gate_reanchor`. So the 0/169 is almost entirely 11-05 to 11-14.

## 4. Does a chain take longer to verify?

Script: `/tmp/gate-10069/measure/item4.py`. Passed verifies only, gate re-verifies excluded.

**How the ratio is computed.** Within each (UTC day, runner) stratum, take each chain's log duration minus the stratum's median single log duration. The ratio is exp of the median of those differences. The confidence interval is a 2000-sample bootstrap.

| window | runner | chain size | n | strata | chain/single ratio p50 [95% CI] | ratio p90 |
|---|---|---|---|---|---|---|
| a | laptop | all | 46 | 12 | 0.97 [0.95, 1.00] | 1.15 |
| a | local | all | 35 | 12 | 1.01 [0.92, 1.07] | 1.34 |
| a | both | 2 / 3 / 5+ | 13 / 22 / 44 | 9 / 13 / 21 | 0.98 / 0.99 / 0.97 | 1.06 / 1.25 / 1.36 |
| a | both | all | 81 | 24 | **0.98 [0.95, 1.01]** | 1.25 |
| b | both | all | 9 | 2 | 0.84 [0.55, 1.14] (**thin**) | 1.14 |

Raw minutes in window (a), passed only, p50/p90:

| runner | size 1 | size 2 | size 3 | size 5+ |
|---|---|---|---|---|
| laptop | 20/40 (n=217) | 21/22 (7) | 37/55 (14) | 22/41 (23) |
| local | 40/66 (n=159) | 32/36 (6) | 45/50 (8) | 42/63 (21) |

Window (b), minutes p50:

| runner | size 1 | size 3 | size 4 |
|---|---|---|---|
| laptop | 65 (n=3) | 39 (2) | 74 (3) |
| local | 137 (1) | 75 (1) | 114 (3) |

- Load drives duration far more than chain size. The daily median for a laptop single went from 17 min on 11-05 to 65 min on 11-19 in window (b).
- Chain build time is negligible: 0.01–0.09 h per day in total.

## 5. Does the rule "admit the n-th member while q < 1/n" bind?

Script: `/tmp/gate-10069/measure/item5.py`.

**Counting rule.** With a single shared q, the rule's largest batch is n_max = the largest n with n·q < 1. The rule binds when n_max is smaller than the size actually verified.

- **Window (b): the table's choice is recorded.** q = 1 − `chain_depth_reason.pass_estimate`. The rule **binds 0 of 13 times, and never ties.** The table chose 3 or 4 members (estimate 0.859–0.903). The rule would have allowed 7 to 10. This agrees with the earlier simulation's "never".
- **Window (a): no table existed.** Depth came from `chain_cap=6` plus halving after a red chain. I rebuilt q at each chain's start in two ways:

| how q was rebuilt | binds | ties | n | red rate when it binds | red rate when it does not |
|---|---|---|---|---|---|
| E1: replay of the 8528 estimator (as in `chain_policy.py::seed_from_rows`): last 30 verdicts, P^(1/mean size) | 62 | 20 | 179 | 37/62 = 0.60 | 61/117 = 0.52 |
| E2: failure rate of the last 30 non-gate singles | 89 | 19 | 179 | 53/89 = 0.60 | 45/90 = 0.50 |

**Under-specified against the schema.** "The depth the table actually chose" exists only in (b). For (a) I compared the rule with the depth that was actually verified.

## 6. Red chains and what followed them

Scripts: `/tmp/gate-10069/measure/item6.py` and `item6b.py`. Member detail exists only from 11-15T11:19Z.

**Red chains by period, red/total (window (a) is the first three rows):**

| period | size 2 | size 3 | size 4 | size 5 | size 6 | total |
|---|---|---|---|---|---|---|
| a1: 11-05 to 11-15T11:19 (no member detail) | 5/18 | 13/23 | 2/4 | 3/10 | 21/40 | 44/95 |
| a2: 11-15T11:19 to 10617 deploy | 1/1 | 20/21 | 1/1 | 1/1 | 22/30 | 45/54 |
| a3: 10617 deploy to 8528 deploy | – | 2/13 | – | 0/2 | 7/15 | 9/30 |
| b: after 8528 | – | 1/4 | 3/9 | – | – | 4/13 |

**What each member's next verify was:**

| period | red chains | members | solo pass | solo fail | chained again before any solo | landed or no later verify | chains with no solo at all |
|---|---|---|---|---|---|---|---|
| a2 | 45 | 203 | 12 | 10 | **179** | 2 | 27 |
| a3 | 9 | 48 | 7 | 5 | **31 (all 31 crossed an orchestrator restart)** | 5 | 2 |
| b | 4 | 15 | 2 | 0 | 0 | 13 (still queued when the data was cut) | 2 |

**Red chains with zero members failing alone:**
- Strict rule (every member got a solo verify): a2 had 1 such chain, with 2 bad members. a3 and b had 0. **So this is effectively unmeasurable.**
- Loose rule (at least one solo verify, and none of them failed): a2 12, a3 3, b 2.

**Eventual attribution.** A member counts as good if it later passed anywhere or landed. It counts as bad if its first later solo failed, possibly after its content changed. Distribution of bad members per fully resolved chain:
- a2 (41 of 45 resolved): 0 bad: 6 · 1: 5 · 2: 22 · 3: 6 · 4: 1 · 5: 1.
- a3 (4 of 9 resolved): 0 bad: 1 · 1: 1 · 2: 2.
- b: none resolved yet.

Two members dominate a2: task 4991 sits in 29 of the 45 red chains and task 7736 in 23 (31 chains contain at least one of them).

**Verify time spent isolating, per red chain** (completed member solos before any re-chain):

| period | total min | mean | p50 | p90 | cancelled member verifies before their next completed verify |
|---|---|---|---|---|---|
| a2 | 900 | 20 | 0 | 57 | 35, 879 min |
| a3 | 662 | 74 | 50 | 152 | 16, 226 min |
| b | 129 | 32 | 32 | 65 | 1, 43 min |

## 7. Waiting set at each pick

Script: `/tmp/gate-10069/measure/item78.py`.

**Recorded** = length of `merge_dequeued.data.waiting`. It never includes the picked request (0 of 209 rows), and exists from 11-15T11:01Z. **Rebuilt** = the earlier study's replay rule from item 7 of `measure.md`. Where both exist, the recorded size is larger by a median of 7 (p5 −1, p95 +18), so I treat the recorded list as authoritative.

| day | picks | recorded p50 / p95 | rebuilt p50 / p95 | heartbeat depth p50 / p95 |
|---|---|---|---|---|
| 11-12 | 76 | – | 6 / 12 | 11 / 15 |
| 11-13 | 62 | – | 7 / 14 | 10 / 18 |
| 11-14 | 88 | – | 12 / 21 | 18 / 26 |
| 11-15 | 45 (28 recorded) | 28 / 31 | 13 / 32 | 16 / 35 |
| 11-16 | 52 | 32 / 33 | 24 / 25 | 35 / 36 |
| 11-17 | 64 | 57 / 67 | 43 / 57 | 59 / 66 |
| 11-18 | 37 | 59 / 85 | 48 / 70 | 59 / 84 |
| 11-19 (a) | 9 | 72 / 75 | 70 / 71 | 77 / 80 |
| 11-19 (b) | 19 | 73 / 79 | 72 / 78 | 76 / 83 |

## 8. Verifies per landing and landings per verify-hour

Same script as item 7. Landings are `merge_finalized` events with state `done`, counted per request. Verify-hours are completed verifies' duration. The second rate also adds the elapsed time of cancelled verifies.

| day | landings (via chain) | completed verifies (gate) | cancelled | verify-h (cancelled h) | verifies per landing | landings per verify-h | landings per h incl. cancelled |
|---|---|---|---|---|---|---|---|
| 11-12 | 52 (24) | 80 (14) | 25 | 35.3 (3.4) | 1.54 | 1.47 | 1.34 |
| 11-13 | 52 (16) | 84 (20) | 19 | 35.4 (1.0) | 1.62 | 1.47 | 1.43 |
| 11-14 | 58 (26) | 74 (4) | 25 | 35.1 (5.6) | 1.28 | 1.65 | 1.43 |
| 11-15 | 44 (38) | 43 (1) | 25 | 26.4 (12.7) | 0.98 | 1.67 | 1.12 |
| 11-16 | 42 (30) | 50 (0) | 26 | 34.8 (7.9) | 1.19 | 1.21 | 0.98 |
| 11-17 | 23 (15) | 47 (1) | 30 | 34.3 (12.4) | 2.04 | 0.67 | 0.49 |
| 11-18 | 48 (42) | 50 (1) | 27 | 35.9 (5.9) | 1.04 | 1.34 | 1.15 |
| 11-19 (a) | 11 (9) | 9 (0) | 7 | 7.5 (1.8) | 0.82 | 1.47 | 1.18 |
| 11-19 (b), about 16 h | 34 (33) | 17 (0) | 9 | 22.5 (4.2) | 0.50 | 1.51 | 1.27 |

## Independent re-check

I recomputed item 1 and the chain totals in pure SQL, without the Python cache: `/tmp/gate-10069/measure/verify.sql`. It uses `json_extract` and marks a gate re-verify with `sha IN (merge_reverify_gate shas)`. Results:

| window | population | SQL result |
|---|---|---|
| a | singles without gate re-verifies | 376/468 |
| a | gate re-verifies | 169/169 |
| a | chains | 81/179 |
| b | singles | 4/4 |
| b | chains | 9/13 |

All match the script figures exactly.

## What I could not measure and why

- **Window (b) is about 16 h of one run.** n is 4 singles, 13 chains, 0 failed singles and 4 red chains, none of them resolved. Every (b) figure is too thin to use.
- **Exact "same content" check for item 3.** `merge_verify` has no branch head sha, and `merge_queued.head_sha` is filled on 44 rows, none of them in a failed→next pair. Only the R1, R2 and git bounds are possible.
- **Red chains before 11-15T11:19.** `chain_task_ids` is absent, so members cannot be identified. Item 6 covers 58 of the 102 red chains. Item 1's isolation-solo tag misses solos after earlier red chains.
- **Pure false failure vs interaction in a red chain.** A member that passes alone after a red chain is either a flake or an interaction, and nothing recorded separates the two. Only 1 red chain was ever fully isolated.
- **Item 5 in window (a).** No table depth was recorded, and no per-request q is recorded in either window. So the rule can only be tested with one shared q, not per member.
- **Cancelled verifies** have elapsed time only (partial work) and no outcome.

## Surprises

1. **The 10617 un-chained mark does not survive a restart.** It lives only in memory: `orchestrator/src/orchestrator/merge_lane/worker.py` sets `_next_dispatch_unchained` to an empty set in `__init__`. All 31 members re-chained before their solo in a3 crossed one of the restarts. Before 10617 (a2), 179 of 203 red-chain members were re-chained without a solo.
2. **The waiting set grew about tenfold, from p50 6 on 11-12 to 73 on 11-19.** The earlier study's "p50 6, p95 13" is stale. The table's recorded `queue_depth` (60–73) agrees with the recorded list.
3. **Gate re-verifies nearly stopped completing after 11-14** (about 1 per day), while `merge_reverify_gate` events continued (2–13 per day), most cancelled with `gate_reanchor`. The 0/169 false-failure bound comes from 11-05 to 11-14.
4. **Chain pass rate does not fall with size the way independence predicts.** Size-3 chains do worse and size-5+ better (z = −2.6 and +2.8). One per-request rate does not describe the data. Two members, 4991 and 7736, were in 31 of the 45 red chains in a2.
5. **Laptop singles pass less often than local ones** (0.781 vs 0.837). Single pass rate fell to 0.24 on 11-15, the only day where completed verifies outnumbered landings so little (0.98) while cancelled verify time hit its peak (12.7 h).

All scripts are in `/tmp/gate-10069/measure/`. No repo file was edited.
