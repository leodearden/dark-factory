# Write-triage M3 toward 0.10: what hides the corrections, and what the levers cost

Task 7519. This document studies the M3 levers and proposes follow-ups; it
changes no production behaviour. Leo ruled on 2026-10-26 (esc-7301-1) that
task 3131 releases at an M3 population point estimate of at most 0.35, with
0.10 as the long-term target. Task 7813 re-confirmed 0.35 under brief v2.2.
This document asks what it would take to reach 0.10. It classifies every
hidden correction in the 2026-10-19 soak and replays the soak corpus through
the shipped judge under each lever. It reports each lever's M3, what the
lever costs in false contests, calls and latency, and a recommended
sequence. Each follow-up is written out in §6 for Leo to authorise. None is
filed.

Every figure re-derives from git alone:
`fused-memory/tests/test_write_triage_m3_levers_committed.py` re-scores the
committed report from the committed arm rows. The figures are in
`fused-memory/calibration/write_triage_m3_levers_report.json` and `.md`.

## 1. The target arithmetic

M3 is the share of CONFLICT writes (writes that dispute their parent) that
end restated, amended, or stored with no mark on the record they dispute
(`fused-memory/scripts/score_write_triage_reader.py::HIDDEN_ENDINGS`). It is
scored under rubric v2.1, the brief the 0.35 and 0.10 bounds were set on.
v2.2 left every CONFLICT answer and M3 unchanged, at 0.379.

Each item is weighted by its stratum, N_h / n_h.

| Stratum | Population N_h | Sampled n_h | Weight | CONFLICT items | CONFLICT population | Soak hidden | Hidden population |
|---|---|---|---|---|---|---|---|
| amended-sample | 1,224 | 100 | 12.24 | 14 | 171.36 | 14 | 171.36 |
| restated-sample | 125 | 100 | 1.25 | 25 | 31.25 | 25 | 31.25 |
| contested-census | 473 | 473 | 1.00 | 332 | 332.00 | 0 | 0 |
| **all** | | | | **371** | **534.61** | **39** | **202.61** |

So the soak M3 is 202.61 / 534.61 = 0.379.

- **0.35** allows 187.1 population hidden writes.
- **0.10** allows 53.46.

Reaching 0.10 means taking at least 149.15 population hidden writes off the
soak's 202.61. Each write that leaves the hidden set moves M3 by:

- an amended item: 12.24 / 534.61 = 0.0229;
- a restated item: 0.0023;
- a contested-census item: 0.0019.

This works in both directions. A contested CONFLICT write that a lever turns
into an amendment, or that the attach guard stores unmarked, adds 0.0019.

The amended stratum decides it. With the restated stratum left as it is, 0.10
needs at least 13 of the 14 amended hits fixed: (202.61 − 13 × 12.24) /
534.61 = 0.081, while 12 fixed leaves 0.104. With every restated hit fixed
too, 10 of the 14 amended hits are still needed: (171.36 − 10 × 12.24) /
534.61 = 0.092.

False contests weigh the same way. A correct amendment that a lever contests
adds 12.24 population false contests. Each of the 86 correct amendments in the
sample therefore stands for about 12 writes of the soak window that would
show a spurious dispute.

## 2. Why the 39 soak hits were hidden

`score_write_triage_m3_levers.py classify` splits the 39 hits by their soak
facts, their rater answers, their replay features and four hand labels
(`fused-memory/calibration/write_triage_m3_hidden_conflicts.jsonl`). The hand
labels are kind of dispute, self-declared correction, and whether the dispute
is in the entry's core. Each label's evidence quotes only a committed
`conflict_quote`. The dispute kinds are defined one sentence each in
`score_write_triage_m3_levers.py::DISPUTE_KINDS`. Population weights are out
of 202.61.

| Dimension | Concentration (sample n; population) |
|---|---|
| stratum | amended 14 (171.36, 85%); restated 25 (31.25) |
| rater SCOPE | secondary 25 (174.12, 86%); central 13 (27.24); tied 1 |
| rater HOME | NO 16 (140.89, 70%); YES 23 (61.72) |
| dispute in the entry's core | no, an aside 15 (117.66, 58%); yes 24 (84.95) |
| dispute kind | status_overtaken 8 (64.95); outdated_state 11 (57.71); newer_measurement 11 (46.72); contradiction 9 (33.23) |
| self-declared correction | yes 7 (19.74, 10%); no 32 (182.87) |
| soak route | judge band 15 (172.61); restate band 23 (28.75); unknown 1 |
| deployed route today | judge 32 (193.86); force-stored 5 (6.25); supersession 2 (2.5) |
| cos_parent | 0.60–0.70: 5 (39.22); 0.70–0.80: 8 (75.94); 0.80–0.90: 17 (65.21); ≥ 0.90: 7 (8.75); < 0.60: 1 (12.24); unknown 1 |

The 14 amended hits carry 85% of the weight, and they share one shape:

- 13 dispute a secondary part of their parent;
- 11 are under a parent whose subject is not the entry's (HOME NO);
- 10 make the dispute in an aside, not in the entry's main claim;
- 12 dispute a time-sensitive detail the parent records: a status (5 are
  status_overtaken, such as "not decomposed" or "no tasks filed" overtaken
  by a later session's progress), a measured figure (3), or where something
  lives (2 more).

Only one of the 14 says it corrects anything.

No feature known at triage time separates these hits from the stratum's 86
correct amendments:

| Feature | Amended hits | Correct amendments |
|---|---|---|
| HOME NO | 11 / 14 | 55 / 86 |
| via sibling | 3 / 14 | 21 / 86 |
| cross category | 3 / 14 | 29 / 86 |
| cos_parent 0.70–0.80 | 6 / 14 | 46 / 86 |

The hits' cosines spread like the controls'.

The 25 restated hits are lighter (1.25 each) and different in kind. 20 are
under their own subject (HOME YES). 8 are later measurements of a figure
their parent records, 7 of them suite timings. 5 claim that a tool does not
exist, against a parent that says it works.

Three conclusions name the levers:

- **No transform over triage-time features.** One that contests the amended
  hits would contest about 6 correct amendments for each hit it catches,
  each worth 12.24 population false contests.
- **No routing of self-declared corrections.** At most 19.74 population
  (about 0.037 of M3) is within reach, only 1 amended hit among it, and it
  would need a text detector.
- **The judge has to read the details.** The lever must make it notice a
  dispute of a secondary, time-sensitive detail, made in passing, by an
  entry that is mostly about something else. That is levers a1 and c1 as the
  task stated them, and lever d1 as the classification names it (§4).

## 3. Method

**One-candidate replay through the shipped judge.** The soak never recorded
the judge's width-5 slates, so production slates cannot be rebuilt. Each case
is the exact (entry, parent) text pair its raters rated: the rebase scratch's
`texts.json`, never committed. Each case goes to the SHIPPED
`fused_memory/server/write_triage_judge.py::judge_write` with a one-candidate
slate of the rated parent. Rater answers are about that pair and blind to the
outcome. A replayed outcome is therefore substituted into the committed item
row and re-scored with the committed verdicts by
`score_write_triage_reader.py::score_reader` on the committed strata.

The runner is `fused-memory/scripts/run_write_triage_m3_levers.py`. A lever
prompt is held in force by
`fused-memory/scripts/write_triage_judge_wording.py::judge_system_prompt`.
No production module was edited.

**Deployed routing, not the soak's.** Each case is routed the way
`fused_memory/server/write_triage.py::triage_write` routes it today
(`fused-memory/scripts/write_triage_m3_replay_cases.py`):

- A supersession is stored with its target marked (F4); 12 cases.
- Caller-owned attach keys or an orchestrator completion record force-store
  the write; 68 cases, 65 of them completion records.
- A restate-band write that is near-verbatim to its target is restated
  without a call (`write_triage.py::route_band`); 2 cases.
- Everything else is judged (591 cases). A judged restatement that is not
  near-verbatim is filed amended, as
  `write_triage.py::_file_judged_restatement` files it (task 7155); that
  refiles 3 to 5 writes per arm. The outcome then passes the attach guard:
  `AttachGuard`, min parent cosine 0.60, hoist gap 0.05.

A guard refusal or a force-store on a CONFLICT write ends stored_unlinked,
which M3 counts as hidden. Caller metadata comes from the soak bundle's
stored records and was recovered for all 673 cases. 4 cases have no measured
cosine; they keep their attach.

**Full-corpus arms.** Every arm replays all 673 items, the 39 hits and every
control, so the false-contest side is measured on the roughly 600 writes the
analyst did not read item by item.

**Controls on the harness.**

- `psi-repro` re-runs the soak's own ψ prompt and model on the soak's
  judge-band items. Its agreement with the soak outcomes is the harness's
  own error: one-candidate slate against width 5, and run-to-run variation.
- `f3p-repeat` re-runs the baseline `f3p` (the shipped F3′ prompt). Its
  verdict agreement with `f3p`, and its M3 against `f3p`'s, are the noise
  floor every lever delta is read against.

**Paired deltas.** Every arm shares one sample, so separate intervals overlap
widely. The decision-grade figure is the paired delta: a stratified bootstrap
of (hidden in arm − hidden in baseline) over CONFLICT writes
(`fused-memory/scripts/write_triage_stats.py::stratified_bootstrap95`, the
scorer's resamples and seed).

**HOME-tie closure.** 29 contested CONFLICT items have a tied HOME, which is
harmless while they stay contested. An arm that turns one into an attach would
make the scorer refuse the corpus. The arm is then scored twice, closing each
tie once each way with an in-memory copy of one tied voter's row. M1, M3, M5
and M2-contested must agree across the two closures, and do, since M3 never
reads HOME on a CONFLICT write. M4, M2-amended and harm are reported as
[low, high].

**In-sample guard.** The same 39 hits motivate and score the levers. The lever
prompts were drafted from the dispute kinds above and from the rater brief's
CONFLICT question, never from a case's text. Every exemplar is synthetic, on
the shipped greenhouse candidate. The runner refuses any prompt sharing an
8-token shingle with a case text
(`run_write_triage_m3_levers.py::refuse_corpus_text`); all arms pass.

## 4. The levers, replayed

Arms, in `run_write_triage_m3_levers.py::ARMS`:

- **f3p**: the shipped F3′ prompt, `gpt-6.1-sol` at low effort. This is the
  baseline every delta is read against.
- **a1-partial-dispute**: the amends/contests boundary moved. An entry that
  adds to a candidate AND disputes any part of it, central or secondary,
  contests.
- **c1-dispute-marker**: a second call over f3p's attached amendments and
  restatements (0.19 extra calls per write). It asks only whether any part
  of the entry disputes the candidate. A yes marks the write, and a marked
  write reads as contested.
- **d1-stale-detail-check**: lever d, named by §2. The shipped prompt plus a
  check of each candidate's time-sensitive details: a status, a location, a
  measured figure, an absence or a limit, or a defect stated as current. It
  contests when the entry shows one has changed, even in passing and even
  when the entry is about something else.
- **e1-mark-refused-contests**: found by the replay, not the soak. Under
  every judge lever, many remaining hidden writes are contests that the
  attach guard stored unmarked, because the parent's cosine is below 0.60
  or the hoist gap is too wide. Under d1 that is 62.72 of 106.21. e1 makes
  no call. When the judge contests and the guard stores the write, it keeps
  the write standalone, as the guard wants, and marks the record it
  disputes: F4's shape for supersessions
  (`run_write_triage_m3_levers.py::mark_refused_contest`). Like c1's
  marker, a marked write scores as contested, so its false contests are
  charged.
- **b1-sol-medium** and **b2-sol-high**: the shipped prompt at higher
  reasoning effort.

**Combinations** are replayed, never summed: **d1-e1-combined** and
**a1-e1-combined** apply e1 over d1's and a1's verdicts.

Figures are population rates from
`fused-memory/calibration/write_triage_m3_levers_report.json`. "False
contests" is the population change in false_conflict cells against f3p.
Calls, latency and USD are what running the arm costs, at list price
($2 / $10 per Mtok): they include the calls of the arm it builds on (the
report's `cumulative_cost`). So a transform costs what its parent costs, and
c1 costs f3p's call plus its own second call; the report's `cost` block
gives each arm's own share. M5
equals M2-contested-correct in every arm: under this rubric a contest is
correct exactly when it has CONFLICT. M4 and harm are [low, high] where a
HOME-tie closure was needed.

| Arm | M3 [95%] | ΔM3 vs f3p [95%] | M1 | M5 | M2 amended correct | False contests | M4 | Harm Δ / write | Calls / write | p50 / p95 s | Fail-open | USD / 1k judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f3p | 0.414 [0.310, 0.502] | — | 0.053 | 0.869 | 0.402 | — | 0.429 | 0.117–0.118 | 0.878 | 2.91 / 4.60 | 0 | 2.89 |
| f3p-repeat | 0.395 [0.293, 0.484] | −0.020 [−0.099, 0.056] | 0.042 | 0.870 | 0.407 | +1.00 | 0.420 | 0.101 | 0.878 | 2.86 / 4.63 | 0 | 2.89 |
| a1-partial-dispute | 0.281 [0.186, 0.375] | −0.133 [−0.222, −0.047] | 0.053 | 0.802 | 0.425 | +46.22 | 0.388 | 0.063 | 0.878 | 3.03 / 5.01 | 0 | 3.10 |
| c1-dispute-marker | 0.251 [0.161, 0.341] | −0.164 [−0.255, −0.071] | 0.009 | 0.652 | 0.430 | +161.38 | 0.378 | 0.162 | 1.070 | 3.12 / 7.75 | 0 | 3.49 |
| d1-stale-detail-check | 0.199 [0.122, 0.284] | −0.215 [−0.315, −0.110] | 0.009 | 0.667 | 0.439 | +161.88 | 0.376 | 0.137 | 0.878 | 3.04 / 5.17 | 0 | 3.30 |
| e1-mark-refused-contests | 0.373 [0.266, 0.464] | −0.041 [−0.061, −0.024] | 0.053 | 0.857 | 0.402 | +8.25 | 0.424–0.425 | 0.113 | 0.878 | 2.91 / 4.60 | 0 | 2.89 |
| **d1-e1-combined** | **0.081 [0.043, 0.136]** | **−0.333 [−0.430, −0.224]** | 0.009 | 0.656 | 0.439 | +205.61 | 0.377 | 0.155 | 0.878 | 3.04 / 5.17 | 0 | 3.30 |
| **a1-e1-combined** | **0.214 [0.123, 0.307]** | **−0.201 [−0.293, −0.110]** | 0.053 | 0.795 | 0.425 | +59.47 | 0.382 | 0.053 | 0.878 | 3.03 / 5.01 | 0 | 3.10 |
| b1-sol-medium | 0.438 [0.336, 0.523] | +0.024 [−0.051, 0.103] | 0.075 | 0.834 | 0.392 | +11.99 | 0.427–0.428 | 0.143–0.144 | 0.878 | 3.18 / 6.10 | 0 | 3.06 |
| b2-sol-high | 0.439 [0.337, 0.523] | +0.025 [−0.049, 0.105] | 0.064 | 0.833 | 0.389 | +12.24 | 0.438–0.439 | 0.153–0.154 | 0.878 | 4.08 / 9.71 | 0 | 3.61 |

Reading the table:

- **The harness.** psi-repro agrees with the soak's outcome on 0.763
  [0.726, 0.796] of 577 soak judge-band items. Of the 469 the soak
  contested, it stored 67 and amended 49. The one-candidate slate and
  run-to-run variation move a quarter of the verdicts, so every figure here
  is a delta against f3p, not a production forecast. f3p itself reads
  0.414, not the soak's 0.379. Deployed routing force-stores 27 CONFLICT
  writes unmarked (22 census, 5 restated; completion records and caller
  attach keys), and the guard stores 20 judged contests unmarked (13 census,
  7 restated). F4 marks 12 supersessions.
- **The noise floor.** f3p-repeat agrees with f3p on 0.961 of verdicts,
  and its ΔM3 interval spans 0, from −0.099 to 0.056. A lever whose
  interval stays below 0 moved M3 by more than rerunning the judge does.
  a1, c1, d1, e1 and both combinations do; b1 and b2 do not.
- **Where the gain comes from.** In the sample, against f3p:
  - d1 fixes 8 of the 12 amended hits, 5 restated and 11 census. It also
    contests 12 of the 86 correct amendments, each worth 12.24.
  - a1 fixes 5 amended hits and 10 census, and contests 3 correct
    amendments.
  - e1 fixes 13 census and 7 restated hits, and no amended one.
  - d1-e1 fixes 11 of the 12 amended hits.
  None of these lever arms newly hides a write the baseline showed.
- **Latency and spend.** Every single-call low-effort arm's p95 is 4.6–5.2 s,
  under the release condition's 6.1–13.0 s bracket. b2 reaches 9.71 s for no
  gain. c1 asks twice in sequence on the 19% of writes it re-asks: its
  second call alone is 3.50 s at p50, and the pair reaches 7.75 s at p95,
  inside the bracket. e1 adds nothing to its parent. A prompt lever costs
  $2.9–3.3 per 1,000 judged writes, and c1 $3.49, against f3p's $2.89. Latency was measured under replay concurrency 12 on
  one-candidate slates. Production's width-5 prompt is longer.

## 5. Recommended sequence toward 0.10

1. **e1, the guard mark, first.** It is deterministic and costs no call.
   Its delta is −0.041 [−0.061, −0.024], wholly below the noise floor, for
   +8.25 population false contests and no correct amendment contested. It
   is also the precondition for the prompt levers' full effect: the more
   the judge contests, the more contests the guard stores unmarked. Under
   d1, e1 is worth a further 0.117 (0.199 → 0.081). Left alone, it leaves
   M3 at 0.373.
2. **a1, the boundary prompt, next.** The replayed a1-e1 reads 0.214
   [0.123, 0.307], comfortably inside 3131's 0.35 bound. It costs +59.47
   population false contests (3 of 86 correct amendments) and has the
   lowest harm of any arm, 0.053 against f3p's 0.117. This is the step to
   ship and then confirm out of sample.
3. **d1, the stale-detail check, only if Leo accepts its price.** The
   replayed d1-e1 is the only arm below 0.10: 0.081 [0.043, 0.136], ΔM3
   −0.333 [−0.430, −0.224]. Its gain over a1-e1 is 0.132 of M3, bought with
   +146 more population false contests. In the sample that is 14 of 86
   correct amendments contested, against a1's 3. M5 falls to 0.656, and
   harm rises to 0.155 under the provisional weights. Its interval's upper
   end, 0.136, does not yet clear 0.10, and its prompt was designed on
   these 39 hits. So it is a candidate for the out-of-sample round, not a
   ship decision.
4. **What d1-e1 leaves: 43.49 population hidden writes.**
   - 28.25 are force-stored completion records whose dispute nobody
     checks. Deployed triage force-stores them before any call, so they end
     stored_unlinked. A lever **e2** would ask the judge, for a force-stored
     write only, whether it disputes its nearest record. On a yes it would
     mark that record, still without attaching. On this corpus that is 68
     of 673 writes, about 0.10 calls per write. It was not replayed: the
     soak recorded no slate for those writes.
   - 15.24 are amendments: one amended hit (12.24) and three census ones.

The arithmetic for 0.10 (§1) says the amended stratum must be almost
clean. Of the measured sequences only d1-e1 gets there, with 11 of 12 fixed.
e2 would put about 0.053 of headroom under it.

## 6. Proposed follow-ups (for Leo to authorise; none filed)

**F1. Mark the disputed record when the attach guard stores a judged
contest (lever e1).**
- *Description:* In `fused-memory/src/fused_memory/server/write_triage.py::triage_write`,
  when the judge answers contests and `AttachGuard.refusals` refuses the
  attach, store the write standalone as now, and mark the disputed
  candidate with the mark F4 writes for a supersession, or its contest
  analogue. Today such a write is stored with no mark, which M3 counts as
  hidden. The guard's purpose, not attaching to a low-cosine parent, is
  kept.
- *Files:* `fused-memory/src/fused_memory/server/write_triage.py`, the
  supersession-mark write path it shares with F4, and
  `fused-memory/tests/server/test_write_triage*.py`.
- *Dependencies:* none. F4 (7156) is done.
- *Acceptance:* a unit test shows a guard-refused contest stored with the
  target marked and not attached. The `e1-mark-refused-contests` replay
  arm, re-run on the shipped path, keeps ΔM3 at or below −0.024 against f3p.
- *Estimated cost:* one implementation task. No judge calls at run time.

**F2. Ship lever a1 as the judge prompt, with the judge accuracy arbiter re-run.**
- *Description:* Replace `JUDGE_SYSTEM_PROMPT`'s amends/contests
  definitions with `run_write_triage_m3_levers.py::A1_PARTIAL_DISPUTE_PROMPT`'s,
  and add its mixed exemplar to `JUDGE_EXEMPLARS`. Record the F3′ prompt
  as a historical wording in `write_triage_judge_wording.py`, so committed
  artifacts keep their era's prompt. Re-run the judge accuracy arbiter
  (`fused-memory/scripts/eval_write_triage_judge.py`, the committed
  `calibration/write_triage_judge_accuracy_report.json`) and the π
  population arms' publish under the new wording.
- *Files:* `fused-memory/src/fused_memory/server/write_triage_judge.py`,
  `fused-memory/scripts/write_triage_judge_wording.py`, and the accuracy
  and population calibration artifacts.
- *Dependencies:* F1, so the gain is not stored unmarked by the guard.
- *Acceptance:* the arbiter's distractor control and attach accuracy are
  no worse than F3′'s. A re-run of `a1-e1-combined` against the shipped
  prompt reproduces ΔM3 ≤ −0.110 against f3p, and false contests within
  +60 population.
- *Estimated cost:* about $2 per corpus replay at list price, plus the
  arbiter's own run.

**F3. Out-of-sample confirmation of M3 after F1 and F2 land.**
- *Description:* Rate a fresh sample as in
  `fused-memory/calibration/write_triage_reader_metrics.md` §7, after the
  levers land, and score it with `score_write_triage_reader.py --strata`.
  The sample needs at least 200 amended writes plus the contested census
  and the stored sample. On the soak, 100 amended items (14 CONFLICT) gave
  a population half-width of about 0.10 at M3 0.38 and 0.05 at 0.08.
  Doubling the stratum narrows it by about √2, to about 0.07 at 0.35, which
  tells 0.10 from 0.35 at 95%. The same round re-scores d1-e1's prompt
  out of sample, by replaying it on the new pairs. That settles step 3 of
  §5 on data the prompt was not designed on. It can ride with task 3131's
  post-fix window.
- *Files:* the strata, items and verdicts of a new rating round, and a new
  baseline report.
- *Dependencies:* F1, F2, and the 3131 window.
- *Acceptance:* a population M3 point estimate with its interval. The
  d1-e1 replay's paired delta on the new pairs is reported against a1-e1.
- *Estimated cost:* one rating round, two blind Opus raters per item, and
  about $4 of replays.

**F4. Lever e2: check force-stored completion records for a dispute.**
- *Description:* For a write `triage_write` force-stores (a completion
  record or caller attach keys), ask the judge, under a c1-style prompt,
  whether it disputes its nearest record. On a yes, mark that record and
  still store standalone. This prototypes the 28.25 population (about
  0.053 of M3) that d1-e1 leaves.
- *Files:* `fused-memory/src/fused_memory/server/write_triage.py`, and a
  replay arm in `run_write_triage_m3_levers.py`, which needs the
  force-stored writes' nearest records: the soak recorded none.
- *Dependencies:* F1, for the mark path.
- *Acceptance:* the replay shows the force-stored CONFLICT writes marked,
  with false contests reported.
- *Estimated cost:* about 0.10 judge calls per write.

**F5. Re-derive this study after the embedder cutover (task 6807).**
- *Description:* The attach guard's 0.60 floor, the hoist gap and every
  cosine bucket in §2 move with the embedder, and with them e1's reach.
  Re-run `write_triage_m3_replay_cases.py build` on re-embedded cosines,
  replay f3p and the shipped lever arms, and re-score this report.
- *Files:* `fused-memory/calibration/write_triage_m3_replay_cases.jsonl`,
  the arm rows, and the report.
- *Dependencies:* 6807.
- *Acceptance:* the committed-report test passes on the re-derived files,
  and the doc's figures are updated.
- *Estimated cost:* about $14 for all arms at list price.

## 7. Caveats

- **One-candidate slates.** Production shows the judge five candidates. The
  replay shows only the rated parent, so slate competition and retargeting
  (an M4 matter) are left out. psi-repro's 0.763 agreement is the size of
  that gap plus run-to-run noise. Read the figures as deltas.
- **In-sample prompt design.** The levers were drafted from the dispute
  kinds of the 39 hits they are scored on. The shingle guard keeps any
  case wording out of the prompts, and the false-contest side is measured
  on about 600 controls. Still, only F3's fresh sample settles M3,
  especially d1's.
- **14 amended items carry 85% of the weight.** Each is worth 12.24 writes,
  so one amended item flipping either way moves M3 by 0.023. Intervals are
  wide for that reason.
- **Judge nondeterminism.** The repeat arm moved 3.9% of verdicts and M3 by
  −0.020. A difference between two arms' M3 is evidence only through its
  paired interval. The repeat arm's own runs from −0.099 to +0.056.
- **Latency under replay.** Measured at concurrency 12 on short
  one-candidate prompts, so production latency is higher. It is not
  comparable one-to-one with the soak's bracket.
- **No stored stratum.** The soak sampled no stored writes against their
  nearest records. So M3 here covers only the attaches and the contested
  census. A CONFLICT write stored standalone at write time is outside every
  figure, and e2's reach is estimated only on the force-stored items.
- **Rubric.** Scored under v2.1, where the bounds were set. v2.2 leaves M3
  and every CONFLICT answer unchanged. M1 is shown for context: it reads
  LOSS, and would read lower under v2.2.
- **Harm weights are provisional** (metrics doc §6), so the harm column
  ranks arms but fixes no price.
