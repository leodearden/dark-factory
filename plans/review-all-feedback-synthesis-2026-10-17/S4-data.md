# S4 data mining: cross-run calibration numbers from the raw review-all feedback packages

Scripts (python3 -I, read-only on inputs): `/tmp/claude-1000/-home-leo-src-dark-factory/313f60b8-8658-4b7a-b926-9c70da52c4ce/scratchpad/review-all-feedback/s4/` (q1.py, q1b.py, q2.py, q3a-d.py, q4.py, q5.py, q6.py, q6b.py).
Labels: **[M]** = measured from a file; **[D]** = derived by my arithmetic from [M]; **[N]** = quoted from narrative/Method prose (not independently measurable); **[I]** = inference.
Run ids: reify = review-all-reify-20261010; scp = review-all-solar_challenge_platform-20261013; sc = review-all-solar_challenge-20261013; df = review-all-dark_factory-20261008.

---

## Q1. Phases 1-4 across the four runs

Sources: each run's `## Method` yaml (first ~130 lines of the .md) and the sibling `.json` (`method.cost`, `method.verification`, `findings[]`, `metrics.totals`). All token figures are the run's own `cost.subagent_tokens_*` (counting rule not stated in any Method block; see Q7-4).

| | reify | scp | sc | df (baseline) |
|---|---|---|---|---|
| src lines (metrics.totals.src_lines / sum of source_lines_by_area) [M] | 880,346 | 59,789 | 26,926 | 531,992 |
| areas / synthesis streams [M] | 9 / 26 | 10 / 26 | 3 (6 area seats) / 28 | 7 / 26 |
| **Seats** (Method `cost.seats`) [N] | opus/high 30 area + 4 cross; sonnet/high 34 skeptics + 25 calibration; fable/high 3 (superseded 1st pass) + 3 (phase-4 rerun) = 99 seat-runs | opus/high 15; sonnet/high 25 (15 skeptics + 10 calibration, [I]); fable/high 3; sonnet/medium filing 1 = 44 | opus/high 10; sonnet/high 11 (**workflow-progress.json shows 10**); fable/high 3 = 24 stated, 23 in workflow-progress.json | opus/high 24; sonnet/high 23; fable/high 3 = 50 |
| Phase 2 sibling-instrument seats [N] | none (stale priors, not re-run) | 11 review + 28 hotspot agents | 32 (22 hotspot + 10 review) | not in Method (SKILL.md: 42 + 10) |
| subagent tokens, Phase 2 [M] | 0 | 7,046,177 | 5,012,450 | 8,634,131 |
| subagent tokens, Phases 3-4 [M] | 25,693,500 | 7,182,872 | 4,254,887 | 14,234,264 |
| subagent tokens, Phase 5 filing [M] | n/a | 179,674 | n/a | n/a |
| subagent tokens, total [M] | 25,693,500 | 14,408,723 | 9,267,337 | 22,868,395 |
| wall clock, phases 1-4 machine time [N] | 7:14 (1:25 + 1:38 + 2:15 + 1:56) over 2026-10-10 21:40 to 10-11 15:48 | 6:50 (0:15 + 2:00 + 4:34); 11:54 elapsed | not stated as machine time; elapsed 16:27. Derived [D]: 0:15 + 1:44 (siblings overlapped) + 4:24 = 6:23 | 8:05 |
| wall clock, longest idle inside it [N] | overnight usage-cap gap (workflow 1 stopped at 33/70 agents) | one usage-cap pause to 23:30 | 14:23-19:41 awaiting lead; plus 100.7 min usage-limit wait inside the 4:24 (Q5) | ~12 h cost-gate wait (outside 8:05) |
| verification confirmed / weakened / refuted / unverified [M] (`method.verification`) | 300 / 47 / 1 / 0 (n = 348) | 76 / 23 / 0 / 0 (n = 99) | 38 / 25 / 1 / 0 (n = 64) | 178 / 25 / 0 / 0 (n = 203) |
| refutation_rate as recorded [M] | 0.003 | 0.0 | 0.016 | field absent (= 0/203 [D]) |
| weakened share of verdicts [D] | 13.5% | 23.2% | 39.1% | 12.3% |
| findings in .json `findings[]` [M] | 359 | 99 | 233 | 203 |
| of which first_seen == this run [M] | 339 | 96 | 59 | 201 |
| high, all findings / high among first_seen == this run [M] | 54 / 48 | 3 / 3 | 28 / 2 | 12 / 12 |
| high share of this-run findings [D] | 14.2% | 3.1% | 3.4% | 6.0% |
| structural / mechanical class [M] | 309 / 50 | 81 / 18 | 164 / 69 | 173 / 30 |
| yellow-flag (calibrate) sample: sampled / confirmed / weakened / refuted [M] | 10 / 5 / 5 / 0 (all high stratum) | 10 / 8 / 2 / 0 (3 high + 7 random) | not run (a skeptic refuted one) | 5 / 5 / 0 / 0 |

Counting rules and traps:
- Verification population differs per run. reify's 348 = all findings the skeptics saw (the .json lists 359 findings: 311 confirmed, 47 weakened, 1 unverified; 339 + 20 hand-converted legacy priors). scp's 99 = every finding. df's 203 = every finding. **sc's 64 = only the skeptic verdicts on this run's new findings** (sum of `seats_findings_verdicts` = 64 = 63 emitted + 1 refuted); the other 170 of its 233 findings are carried priors from the two Phase 2 siblings, which carry their own verdicts (`prior_verdicts` 172) and are not in the rate. So the four refutation_rates are not on one denominator. Pooled new-finding verdicts: 714, refuted 2 (0.28%), weakened 120 (16.8%) [D].
- sc's report text says the critic corrected "refuted 1" to 0 on the claim no finding carried it; the final .json still has `method.verification.refuted = 1` and `refuted[]` has one entry (config-cli seat). [M]
- "this run" = `findings[].first_seen == method.run_id`. sc's 59 is not the 61+2 the synthesis text names (supersession at join); counting rule is mine.
- Tokens per run include waste: reify's 25.69M = 11,177,056 interrupted workflow 1 + 11,310,144 resume (re-ran 5 area seats + 6 skeptics the cache missed) + 3,206,300 phase-4 rerun (sum checks; `skill_port_seat` 228,590 excluded). [M, sum verified]
- Seat-run count for reify (99) counts the superseded fable first pass (3); 96 without.

**Cross-run ratios [D]**

| | reify | scp | sc | df |
|---|---|---|---|---|
| this-run findings per 1M Phase 3-4 tokens | 13.2 | 13.4 | 13.9 | 14.1 |
| this-run high findings per 1M Phase 3-4 tokens | 1.87 | 0.42 | 0.47 | 0.84 |
| Phase 3-4 tokens per src line | 29.2 | 120.1 | 158.0 | 26.8 |
| Phase 3-4 tokens per seat-run (÷ 99 / 43 / 23 / 50) | 259k | 167k | 185k | 285k |
| Phase 2 as share of total | 0% | 48.9% | 54.1% | 37.8% |

---

## Q2. Phase 6 seats, solar_challenge_platform (seats.csv)

Columns used: `session, kind, stream, model, effort, assistant_turns, input_tokens, output_tokens, cache_read, cache_create, approx_subagent_tokens, first_ts, last_ts, degenerate`. `approx_subagent_tokens = input + output + cache_create` (asserted equal on all 63 rows); minutes = `last_ts - first_ts` of the seat transcript (wall, includes waiting inside the seat); raw total = input + output + cache_read + cache_create. 63 rows = 48 (dead lead b1f15ebe) + 15 (resumed lead 13d7dc93). The csv has **no "ruling" kind** (kinds present: prd-author(agent), prd-decompose, clerk, phase5-filing, merge-redrive); the four 2026-10-16 ruling sessions and the spawn-mode authors are fleet sessions, not in this csv.

Non-degenerate rows (`degenerate == False`):

| kind | model/effort | n | tokens median (approx) | tokens range | minutes median | minutes range | turns median | output tok median | cache-read share of raw |
|---|---|---|---|---|---|---|---|---|---|
| prd-author(agent) | opus/high | 13 | 567,274 | 292,620 - 765,751 | 50.4 | 30.8 - 65.7 | 121 | 71,743 | 97.3% |
| prd-decompose | opus/high | 26 | 571,726 | 279,334 - 1,102,862 | 37.8 | 17.9 - 72.8 | 100.5 | 41,782 | 97.1% |
| clerk | sonnet/medium | 16 | 452,681 | 166,377 - 1,207,491 | 11.2 | 5.5 - 33.4 | 32 | 5,993 | 80.8% |
| phase5-filing | sonnet/medium | 1 | 1,340,111 | - | 129.9 | - | 76 | 17,548 | 85.8% |
| merge-redrive | sonnet/medium | 1 | 162,125 | - | 9.8 | - | 25 | 4,381 | 90.9% |

Degenerate rows (6, flag as in csv): decompose S14 / S23 / S20: 24-26 turns, 8.9-9.3 min, 217,408-260,345 tokens each (the first attempts; each stream has a second decompose row starting 15:22:40Z); decompose S17 and S21: 1 turn, 0 tokens, 7-18 s (first_ts 15:11:41Z); clerk (13d7dc93): 14 turns, 4.1 min, 212,508. Degenerate total 947,410 approx tokens = 2.9% of the csv total. 5 of 26 streams (S14, S17, S20, S21, S23) needed two decompose rows.

Totals [D]: approx 32,980,964 (dead lead 23,954,766; resumed 9,026,198); raw 850,556,305; cache read 817,575,341 = **96.1% of raw**. By kind (all rows): decompose 15,360,061 (31 rows), clerk 8,771,318 (17), author 7,347,349 (13), filing 1,340,111, redrive 162,125 (sum checks). Per-seat cache-read per turn medians: author 176.6k, decompose 170.6k, filing 106.4k, clerk 59.2k.

Other measured shapes:
- Per-stream author+decompose cost (26 streams): median 819,395, range 475,176 - 1,522,595 (S26). Agent-authored streams (13): median 1,099,044; streams with a spawn-mode author (13, decompose only in csv): 620,561.
- Author return to first decompose start: median 38.6 min (9.1 - 99.8) across the 13 agent-mode streams; the lead gated the second step.
- Phase 6 subagent tokens in this csv (33.0M, input+output+cache_create rule) are 2.3x the run's Phases 1-5 total (14.4M) [D]; the two figures may not share a counting rule (Q7-4).

---

## Q3. Lead context curves (solar_challenge_platform)

Columns used: `timestamp, context_tokens, output_tokens` (the only three; **no turn-content hint exists in either csv**). Rows are per streamed content block of an assistant message: 457 rows (dead lead) and 255 rows (resumed) collapse to **148 and 93 turns** by grouping rows with identical `context_tokens` (context never repeats across turns; `output_tokens` is constant within a group; context is monotonically non-decreasing, i.e. no compaction in either session). [M]

| | dead lead b1f15ebe | resumed lead 13d7dc93 |
|---|---|---|
| span | 2026-10-13 06:32:59Z to 10-14 16:45:08Z | 10-15 13:46:45Z to 10-16 16:49:59Z |
| turns | 148 | 93 |
| context start / end | 69,130 / 970,432 | 60,300 / 585,024 |
| max | 970,432 (last turn, 16:45:04Z) | 585,024 (last turn) |
| turns with context > 800k | **29 turns (84 rows)**, first 15:20:10Z (805,775), last 16:45:04Z; 84.9 min | 0 |
| first turn at >= 500k / 600k / 700k / 800k / 900k | 12:36Z / 13:49Z / 14:38Z / 15:20Z / 16:02Z (10-14) | >= 300k 14:17Z, >= 400k 15:07Z, >= 500k 20:24Z (10-15) |
| output tokens per turn | median 1,244; mean 2,245; p90 5,165; max 15,602; sum 332,205 | median 1,018; mean 1,994; p90 5,275; max 13,103; sum 185,444 |
| context growth per turn | median 4,168; mean 6,131; p90 13,622; max 36,085 | median 3,079; mean 5,704; p90 16,160; max 39,487 |

Context at identifiable boundaries (turn at or before the timestamp; times Z; boundary events from seats.csv and program-commits.txt, commit times converted from +0100) [M/D]:

| dead lead boundary | time | context |
|---|---|---|
| first turn (phase 0) | 10-13 06:32 | 69,130 |
| Phase 5 filing seat spawned | 10-13 23:22 | 388,393 |
| Phase 5 filing seat returned | 10-14 01:32 | 426,359 |
| report commit e4f2a2b | 10-14 11:46 | 434,021 |
| program doc commit 207d800 | 12:37 | 508,112 |
| first author seat spawned (Phase 6 start) | 12:51 | 525,868 |
| last author return | 13:57 | 612,228 |
| first decompose spawn | 14:06 | 629,202 |
| decompose batch 1 commit 6d51f4c | 14:59 | 773,236 |
| five degenerate decomposes end / respawn x5 | 15:11 / 15:22 | 786,835 / 811,647 |
| first clerk spawned | 16:00 | 889,564 |
| decompose batch 2 commit af1a7fd | 16:05 | 910,838 |
| S5/S6 commit 6791755 | 16:41 | 949,924 |
| last turn (context-limit death) | 16:45 | 970,432 |

| resumed lead boundary | time | context |
|---|---|---|
| first turn | 10-15 13:46 | 60,300 |
| at 14:07 (21 min in) | | 263,063 |
| clerk batch 3-9 start | 14:13 | 288,004 |
| decompose seats start | 14:21 | 314,541 |
| commit b979804 (S2,S7,S11,S12,S16 decomposed) | 15:19 | 427,883 |
| commit b9a80a4 (S15; FILED table complete) | 15:37 | 444,510 |
| merge-redrive clerk | 20:26 | 501,552 |
| first ruling-day commit window (10-16 08:00) | | 518,425 |
| last turn | 10-16 16:49 | 585,024 |

Phase split of the dead lead [D]: phases 0-5 = 78 turns, 69,130 to 525,868 (+456.7k), output 182,372; Phase 6 = boundary to death = 70 turns, 525,868 to 970,432 (+444.6k), output 149,833; Phase 6 growth rate 110k per hour; median gap between turns 1.8 min in Phase 6 vs 1.0 min before.

Growth per seat return. Method: new_input(i) = context(i) - context(i-1) - output(i-1) (what entered the context other than the lead's own previous output; no negative values in either session). A seat "lands" in the first lead turn after its `last_ts`. [D, I: output tokens are assumed to enter the context 1:1]

| | dead lead | resumed lead |
|---|---|---|
| total growth | 901,302 | 524,724 |
| = lead's own output | 331,711 (36.8%) | 180,325 (34.4%) |
| = new input | 569,591 | 344,399 |
| new input in turns where >= 1 seat landed | 130,912 (23.0%) in 31 turns | 48,006 (13.9%) in 16 turns |
| new input in all other turns | 438,679 in 116 turns (median 1,326) | 296,393 in 76 turns (median 1,186) |
| decompose-return turn: new input median (range) | 4,068 (117 - 12,323), 15 turns; 1 turn bundled the 5 degenerate returns: 12,951 | 5,412 (4,134 - 5,938), 6 turns |
| author-return turn | 3,232 (2,211 - 10,286), 10 turns | n/a |
| clerk-return turn | 653 (635 - 668), 4 turns | 1,029 (540 - 5,512), 8 turns |
| lead response lag after a landing seat (median) | 0.9 min | 0.5-0.6 min |
| largest single-turn new inputs | 33,844 (10-14 14:38), 32,380 (10-13 06:46), 23,062 (06:51), 20,921 (10-14 14:56), 19,857 (10-13 11:50); none coincide with a seat return | 37,689 (10-15 14:04), 31,986 (13:58), 23,255 (13:55), 20,466 (13:51), 20,369 (14:07); none coincide with a seat return |

What consumed the most (I, since the csv has no content): not seat returns. The five largest turns of both sessions are lead-side reads with no seat landing; in the resumed lead, its first ~10 turns (13:47-14:07Z) took the context from 60,300 to 263,063 (+202.8k, 38.6% of that session's whole growth) before any seat was spawned, i.e. the resume re-read cost.

---

## Q4. reify Phase 6 timeline (successor lead)

Sources: `derived/stream-commit-timeline.json` (14 rows: 12 spawn-mode, 2 agent-mode; pass-1 and pass-2 commit timestamps, exact, +01:00), `derived/session-timeline.json`, `derived/lead-observation-log.md` (spawn times, **narrative, "approximate"**), `scratch/wave2/roster.md`. The json holds **no spawn time**; the spawn to pass-1 column uses the log's times and is marked [N]-input. The 14 rows are the successor lead's streams; the other 12 of the 26 streams were filed in wave 1 by the original lead and are not in the file.

| stream | mode | pass-1 commit | pass-2 commit | pass-1 to pass-2 (h) [D] | spawn to pass-1 (h) [D from log spawn time] |
|---|---|---|---|---|---|
| S24 | spawn | 10-13 15:14 | 15:38 | 0.39 | 21.3 |
| S6 | spawn | 10-13 16:27 | 17:33 | 1.10 | 22.5 |
| S4 | agent | 10-13 17:46 | 18:09 | 0.38 | 3.6 |
| S8 | spawn | 10-13 20:07 | 20:23 | 0.27 | 3.7 |
| S10 | spawn | 10-14 10:59 | 14:00 | 3.03 | 17.3 |
| S25 | spawn | 10-14 11:12 | 11:56 | 0.73 | 21.0 |
| S5 | spawn | 10-14 11:59 | 13:17 | 1.29 | 21.8 |
| S12 | spawn | 10-14 14:22 | 14:48 | 0.43 | 16.9 |
| S1 | spawn | 10-14 17:55 | 18:51 | 0.94 | 4.5 |
| S26 | spawn | 10-15 12:29 | 12:46 | 0.28 | 21.2 |
| S16 | spawn | 10-15 14:30 | 14:59 | 0.49 | 23.5 |
| S18 | spawn | 10-15 14:37 | 15:02 | 0.41 | 27.5 |
| S23 | agent | 10-15 17:29 | 17:49 | 0.34 | 1.9 |
| S2 | spawn | 10-15 18:08 | 19:20 | 1.20 | 23.2 |

| mode | n | pass-1 to pass-2 h: median (min - max) | spawn to pass-1 h: median (min - max) | spawn to pass-2 h: median |
|---|---|---|---|---|
| spawn | 12 | 0.61 (0.27 - 3.03); sum 10.54 | 21.3 (3.7 - 27.5); 9 of 12 >= 17 h | 21.7 (max 28.0) |
| agent | 2 | 0.36 (0.34 - 0.38) | 2.7 (1.9, 3.6) | 3.9 (S4) and 2.2 (S23) |
| all | 14 | 0.46 (0.27 - 3.03); sum 11.26 | - | - |

- Phase 6 wall clock [D]: first spawn (S24/S6, 10-12 17:57 BST, from the log) to last pass-2 commit (S2, 10-15 19:20 BST) = **73.4 h**; first pass-1 commit to last pass-2 commit (json only) = 52.1 h; the successor lead session (registry) started 10-13 12:58Z and its result file was last written 10-16 14:16Z = 73.3 h (this includes the gap-fill sessions). Sum of the 14 pass-1-to-pass-2 intervals is 11.3 h, 15% of the 73.4 h.
- Registry cross-check [M]: the three spawn sessions present in the registry (S18, S16, S2) have `start_to_result_hours` 28.0 / 24.0 / 24.4, and their start to pass-2 commit is 27.94 / 24.0 / 24.38 h. In 3 of 3 the session's result file lands within about 4 minutes of the pass-2 commit [I: spawn sessions stay open through the lead's pass 2].
- Agent-mode roster (only S4, S23; the roster has no entry for any spawn stream): S4 = 6 invocations 435k + 396k + 549k + 243k + 588k + 666k = 2,877k tokens (+ 76k sonnet count(map) repro), 72+58+21+40+10+19 = 220 seat-minutes; S23 = 467k + 308k + 543k + 634k = 1,952k, 150 seat-minutes. Resumed-seat counts include re-read context per the roster note; counting rule otherwise unstated.
- `session-timeline.json`: **7 sessions registered at capture** (2026-10-16T16:52:05Z): 1 awaiting-input (team lead), 5 exited code 0 (prd S18, S16, S2; hotspot gap-fill; review gap-fill), 1 idle (gapfill-deliberation). The observation log states registry records were reaped within 1-2 days (S24/S6/S5/S25/S8/S10/S12/S1/S26 gone), so 7 is a floor, not the session count; at least 12 spawn-mode PRD sessions existed. [M + N]

---

## Q5. solar_challenge workflow (Phases 3-4, wf_7866a6f5-923)

**The journal does not carry usage.** `journal.jsonl.gz` has 48 lines with fields `type` (launched / started / result), `key`, `agentId`, `label`, `phase`, `result`; no tokens, duration, model or timestamp. The per-agent figures below come from `data/scratch/workflow-progress.json` (fields `tokens, toolCalls, durationMs, startedAt, queuedAt, model, attempt, lastAttemptReason`). Check: sum of `tokens` over its 23 agents = 4,254,887 and tool calls 962 = Method `phases_3_4` exactly. Journal: 24 `started`, 23 `result`: the 24th start is the first critic attempt, which never returned (usage limit); its tokens are not in the 4,254,887 [I].

| phase (kind) | agents | model | tokens sum | tokens median (range) | minutes median (range) | tool calls |
|---|---|---|---|---|---|---|
| Area review | 6 | opus-5-5 | 1,431,214 | 224,718 (195,523 - 300,265) | 24.4 (19.0 - 35.5) | 344 |
| Verify (area skeptics) | 6 | sonnet-5-5 | 551,956 | 90,848 (74,832 - 108,544) | 12.0 (5.7 - 13.3) | 149 |
| Cross-area lenses | 4 | opus-5-5 | 810,546 | 205,076 (177,362 - 223,032) | 33.1 (28.8 - 39.2) | 280 |
| Verify (cross skeptics) | 4 | sonnet-5-5 | 444,565 | 112,943 (94,647 - 124,033) | 17.1 (13.6 - 20.8) | 143 |
| Synthesize | 1 | fable-5-1 | 316,781 | - | 22.9 | 22 |
| Critic (attempt 2, after "throttled") | 1 | fable-5-1 | 494,341 | - | 23.7 (durationMs) | 15 |
| Synthesize:revised | 1 | fable-5-1 | 205,484 | - | 21.2 | 9 |
| Calibrate | 0 | - | 0 | - | - | - (phase declared, no agent: skeptics refuted 1) |
| total | 23 | 10 opus / 10 sonnet / 3 fable | 4,254,887 | | agent-minutes 482 (sum of durationMs 493) | 962 |

Longest single agent by duration: cross:stateless, 39.2 min (area-review longest: simulation-core/physics 35.5 min, 280,725 tokens). Largest by tokens: critic 494,341 (more than 2x the SKILL.md expected 100-200k). Phase labels in the Method are `fable/high` / `sonnet/high`; the progress file records model ids (`claude-fable-5-1`) but not effort.

Timeline (minutes from the first queue time, 2026-10-13 18:41:17Z) [D]:

| step | start | end |
|---|---|---|
| 6 area seats in parallel | 0.0 | 19.0 - 35.5 |
| area verify (each starts when its seat ends) | 19.1 - 35.5 | last ends 47.2 (physics verify, 11.6 min) |
| 4 cross lenses start together | 47.2 | 76.0 / 77.2 / 83.4 / 86.4 |
| cross verify | 76.1 - 86.4 | last ends 106.0 |
| synthesize | 106.0 | 128.9 |
| critic attempt 1 queued | 128.9 | usage limit; attempt 2 starts 229.6 (reset 23:30 London = 22:30Z) |
| critic attempt 2 | 229.6 | 243.0 (last progress) |
| synthesize:revised | 243.0 | 264.2 (= 4:24) |

Long tail or breadth? Measured by event sweep of agent start/end (critic end taken as last progress):
- 100.8 of 264.2 minutes (38%) had no agent running (usage-limit wait, 128.9 to 229.6).
- Active time 163.5 min (2:44). Concurrency-minutes: exactly 1 agent running 73.9 min (45% of active), 2: 2.3, 3: 9.0, 4: 44.0, 5: 5.0, **6: 29.3** (only the area phase reaches 6). Average concurrency over active time 2.95.
- So the 4:24 was neither one monster agent (longest is 39 min) nor breadth: it was **depth plus a pause**: 6 barrier-separated phases (cross lenses did not start until the slowest area seat's skeptic ended at 47.2, though 5 of 6 area verifies had ended by 38.7), a serial tail of synthesize, critic, revise (about 58 min of single-agent time), and a 100.8 min usage-limit wait. Without the pause the run is 2:44.

---

## Q6. solar_challenge filing mechanics (data/scratch/submit.log, submit_retry.log, resolve.log, filings.json)

Neither log has timestamps and the ticket ids do not decode as ULID times with a sane epoch, so **resolve latency in seconds is not measurable from these files**. What is measurable:

| | pass 1 (submit.log) | pass 2 (submit_retry.log) | total |
|---|---|---|---|
| tickets submitted (lines `fk-..: ticket tkt_..`) | 53 | 29 (only those that failed in pass 1) | 82 submissions for 53 findings |
| resolved | 24 (23 created + 1 combined) | 29 (23 created + 6 combined) | 53 (46 created, 7 combined) |
| failed | 29, all `server_restart` (55%) | 0 | |
| resolves whose inner retry hit >= 1 read error | 9 of 24 | 10 of 29 | 19 of 53 (36%) |
| resolves that exhausted the 3-attempt inner retry (log line `resolve attempt 1 failed`, then succeeded on the outer retry) | 4 | 7 | 11 of 53 (21%) |
| retry log lines | 18 (17 ReadTimeout, 1 ReadError) | 20 (all ReadTimeout) | 38 |

- `resolve.log` is the 29 failed-ticket list (`failed server_restart`) written by the repair script `resolve_pending.py` (which re-resolves with a 300 s client timeout); it matches the 29 server_restart lines in submit.log one for one. [M]
- filings.json: 69 rows (step `none` 53, step `2` 13, `lead-weakened` 3); disposition filed 66, open 3. filings_retry.json: 29 rows, all step `none`.
- Median gap between consecutive ticket ids is 0.61 s (pass 1) and 0.65 s (pass 2) if the id is read as a millisecond counter [I, unverified]; this would put the 53 submits in about 62 s and the 29 in about 23 s. Do not quote as measured.
- scp's report states a comparable failure ("25 tickets failed server_restart ... re-run in full; 5 rows resolved by hand after resolve_ticket timeouts"). A fused-memory restart mid-filing hit both runs [N].

---

## Q7. What the data shows that the retrospectives did not say

1. **Findings per Phase 3-4 token is nearly constant across the four runs (observation).** 13.2 / 13.4 / 13.9 / 14.1 this-run findings per 1M tokens (reify / scp / sc / df), despite a 33x spread in source size and a 2x spread in seat sizing. The ratio does not hold for high findings (1.87 / 0.42 / 0.47 / 0.84 per M) [D]. Inference: output scales with seat-minutes, not with code; budget a run by seat count x ~170-285k, and expect ~14 findings per M tokens.
2. **Tokens per source line are not a usable budget (observation).** 26.8 to 158 per line; small repos cost ~5x more per line because area plus four cross lenses are a fixed overhead (sc: 4 cross lenses plus verifies = 1.26M of 4.25M, 30%).
3. **Verification populations are not comparable (observation).** See Q1; the rates 0.0 / 0.003 / 0.016 / absent sit on denominators of 99 / 348 / 64 / 203. The weakened share (12.3% / 13.5% / 23.2% / 39.1%) is a better-behaved signal and moves 3x between runs.
4. **Two token-counting rules coexist (observation + inference).** scp's Method records the Phase 5 filing seat at 179,674 tokens; the same seat in seats.csv (the only sonnet/medium filing seat, 76 turns, 130 min) is 1,340,111 under input+output+cache_create (1.32M of it cache_create) and 9.4M raw. A ~7.5x difference. The Method figure likely comes from the workflow tool's own count [I]. Any cross-run comparison of Phase 6 against Phases 3-4 needs a stated rule; as published, Phase 6 (33.0M) vs Phases 1-5 (14.4M) mixes both.
5. **Yellow-flag samples downgraded 7 of 25 sampled "confirmed" findings (observation).** reify 5 of 10 (all high), scp 2 of 10, df 0 of 5; 0 refuted. In reify the 10 sampled were high-severity only and half were weakened; the headline "1 refuted in 348" hides this. Also reify ran 25 calibration skeptics versus the 10 the skill prescribes (orchestration.md `CALIBRATION_SAMPLE = 10`), 15 more than designed [N+M].
6. **Method seat counts drift from the workflow record (observation).** sc Method says opus 10 / sonnet 11 / fable 3 (24 seats) against `agents_phases_3_4: 23`; workflow-progress.json has 10 opus / 10 sonnet / 3 fable = 23 (sonnet = 6 area verifies + 4 cross verifies). The Method sonnet figure is one too high; the journal's 24th `started` is the lost fable critic attempt, not a sonnet.
7. **The sc critic was the most expensive single agent (494k) and the only one throttled (observation).** It costs 2.5x the skill's 100-200k estimate, and its first attempt's tokens are untracked. The synthesis is 317k vs the 200-300k estimate. Area seats (median 225k) and cross seats (205k) sit in the 150-250k band; skeptics (91k area, 113k cross) in 60-100k+.
8. **Phase barriers, not seat count, set sc's wall clock (observation).** All six area skeptics but one finished by 38.7 min, but cross lenses waited for the physics verify to end at 47.2 min (8.5 min of 4-lens idle); single-agent time was 45% of active time (Q5). Inference: starting a cross lens per completed area, or at least starting the synthesis-input collection earlier, would not change the serial tail but would cut ~8 min.
9. **The dead lead died with 457k already spent before Phase 6 started (observation).** Phases 0-5 took the lead from 69k to 526k (78 turns, 182k lead output); Phase 6 then had 474k of headroom to an assumed 1M limit [I: the limit is not in the data] and used 445k in 70 turns. Seat returns were only 23% of the non-output growth; the largest jumps (20-34k) are lead reads. Orphan effect: three decompose seats (S13 ended 17:05Z, S10 17:31Z, S1 17:47Z) kept running after the lead died at 16:45Z; the resumed lead needed 21.0 h to restart (death 16:45Z, resumed first turn 10-15 13:46Z; the commit gap is 21.6 h) and spent its first 21 min and +203k tokens re-reading state. Inference: the resume itself costs ~200k of context.
10. **Restart waste in the scp decompose wave (observation).** Five decompose seats were restarted at 15:22:40Z after three had run 9 minutes and ended with 217-260k tokens each and two 1-turn zero-token rows; 947k approx tokens (2.9%) plus ~11 minutes of lead wait on the 5-seat bundle. The reason is not in the csv.
11. **Agent-mode vs spawn-mode in reify is a 10x end-to-end difference (observation, n = 2 vs 12).** Agent-mode streams S4 and S23: 3.9 h and 2.2 h from launch to pass 2 (including critic and two folds for S4). Spawn-mode median 21.7 h, with 9 of 12 waiting >= 17 h for pass 1, overnight. Pass-1 to pass-2 is also shorter for agent mode (0.36 h vs 0.61 h median). Cost side: the roster's token counts for the two agent-mode streams (2.9M and 1.95M, rule unstated) are on the same order as scp's agent-mode stream medians (1.10M author+decompose, csv rule).
12. **Spawn sessions close with the lead's pass 2, not with the author's return (observation, n = 3).** Session result time equals pass-2 commit time within 4 minutes in S18 / S16 / S2, so `start_to_result_hours` over-states author work by the lead's wait.
13. **The registry under-counts (observation).** 7 sessions at capture vs at least 12 spawn-mode PRD sessions plus gap-fills; records are reaped in 1-2 days.
14. **sc ticket resolution is flaky even without a restart (observation).** 36% of successful resolves saw a read error and 21% exhausted the 3-attempt inner retry; a mid-filing fused-memory restart took 55% of pass 1. 82 submissions were needed for 53 findings.
15. **scp's 3 high findings of 96 (3.1%) and sc's 2 of 59 (3.4%) against reify's 14.2% and df's 6.0% (observation).** Whether this reflects the code or the seats' severity discipline cannot be told from these files; sc's 28 total highs are mostly carried priors (26).

---

## Data quality note

Could not parse / missing:
- `journal.jsonl.gz` (sc): parsed (48 lines), but has **no usage, tokens, duration, model or timestamp** fields; per-agent figures taken from `workflow-progress.json` instead. First critic attempt (agent a074f0f64b4e5c23d) has a start and no result and no usage anywhere.
- `workflow-progress.json`: critic `durationMs` (23.7 min) exceeds start to last-progress (13.4 min); revised synthesis starts at the critic's last-progress time (243.0), so I used 13.4 for the sweep and show 23.7 in the table. Effort levels not recorded.
- `submit.log` / `resolve.log` / `submit_retry.log`: no timestamps; ticket ids do not decode as standard ULIDs; **no resolve latency in seconds**. Only counts and retry lines.
- `stream-commit-timeline.json`: no spawn time; spawn to pass-1 uses `lead-observation-log.md` times, which the log calls approximate (BST assumed). Covers 14 of reify's 26 streams.
- `roster.md`: only the 2 agent-mode streams (S4, S23); one S23 critic line is "running" with no figures (skipped); token counting rule unstated.
- `lead-context-*.csv`: only timestamp, context_tokens, output_tokens; no turn content, no message ids (turns inferred by identical context_tokens). `seats.csv`: no ruling-kind seats; spawn-mode (fleet) sessions absent.
- dark-factory Method block has no `refutation_rate`; reify/scp/sc Method token counting rule is unstated; scp Method gives only model/effort counts, not per-seat sizes.
- Not examined (out of scope): scp's own `filings.json` dispositions; reify Phase 2 (none); transcripts.

Reproduction (one figure per table re-run with an independent method, all matched):
- Q1: reify this-run findings 339 = `grep -o '"first_seen": *"review-all-reify-20261010"' ...json | wc -l`.
- Q2: median approx tokens of the 26 non-degenerate prd-decompose rows 571,726 (range 279,334 - 1,102,862) via awk + sort; csv totals 32,980,964 approx / 850,556,305 raw / cache read 96.12% via awk.
- Q3: unique context values 148 and 93 via `cut | sort -un`; 29 above 800,000 in the dead lead.
- Q4: S18 spawn to pass-1 27.55 h and S2 pass-1 to pass-2 4,306 s = 1.20 h via GNU `date`.
- Q5: sum of per-agent tokens 4,254,887 via `jq`.
- Q6: 29 `server_restart`, 53 + 29 ticket lines, 4 + 7 `resolve attempt 1 failed`, 17 + 20 ReadTimeout via `grep -c`.
