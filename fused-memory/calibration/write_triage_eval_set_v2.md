# Write-triage eval set v2: every top-5 judge pair, double-rated

This is the reference for the fully labelled write-triage evaluation set built
by task 7160. Every (write, record) pair that production's judge would be shown
at width 5 is rated blind by two Opus raters under the rubric of
`write_triage_reader_metrics.md` §4. The labels come from the rater brief
`write_triage_rater_brief_v2.md`. That is v2.1 for HOME, CONFLICT, SCOPE and
action_changing, and v2.2 for every LOSS answer the set consumes (§6). The set
has a frozen, hash-pinned held-out split.

Every figure here is quoted from the committed JSON files named in §2, or
counted from the committed JSONL beside them, unless it is marked otherwise.
Rates and kappas are quoted as the JSON gives them.

## 1. What it is for

The set is a fixed, labelled corpus of judge inputs. It scores a judge
configuration by what a later reader of the corpus meets, without a new soak.
Its consumers:

- **Judge prompt variants**: score a variant's answers on the tuning split, and
  on the test split only as §9 allows.
- **The granite re-measure after 6807**: `write_triage_reader_metrics.md` §7
  re-runs the measurement after the embedder cutover. The texts and labels here
  do not depend on the embedder. The slates do (they are cosine top-5), so
  6807 needs new slates for any new pair, rated the same way.
- **The reconciliation replay eval**, and any later gate that needs labelled
  judge pairs.

## 2. Files, and how to re-derive each one

All paths are under `fused-memory/calibration/`. Run the commands from
`fused-memory/`. `S` is `/home/leo/src/dark-factory/data/write-triage-population-2026-10-05/snapshot.json`,
which is gitignored, read-only, and only in the main checkout.

| File | What | Re-derive |
|---|---|---|
| `write_triage_eval_set_v2_live_slates.jsonl` | header + 400 rebuilt live judge-band writes (§4) | `scripts/rebuild_write_triage_live_slates.py` (its docstring gives the command) |
| `write_triage_eval_set_v2_rounds/key.json` | the scratch key: shard items, raters, rounds, the `rerates` record | `scripts/build_write_triage_eval_set.py shards`, `reask`, `third`, `rerate` |
| `write_triage_eval_set_v2_rounds/ratings/*.jsonl` | 238 rater files, one per shard, 9,367 lines | written by the blind raters; never edited |
| `write_triage_eval_set_v2_writes.jsonl` | 884 write rows, each with its ranked targets | `build_write_triage_eval_set.py assemble --snapshot S --out-dir calibration/` |
| `write_triage_eval_set_v2_verdicts.jsonl` | 8,733 effective rater rows, DIGEST added | the same `assemble` |
| `write_triage_eval_set_v2_summary.json` | pairs, labels, kappa, rating volume, re-rates, shards, strata, input hashes | the same `assemble` |
| `write_triage_eval_set_v2_split.json` | the frozen split (§9) | `build_write_triage_eval_set.py split`; refuses to overwrite without `--refreeze` |
| `write_triage_eval_set_v2_cross_family.{jsonl,json,md}` | the cross-family re-rate (§10) | `scripts/rate_write_triage_cross_family.py rate`, then `report` |

The round commands take `--snapshot S --live-slates calibration/write_triage_eval_set_v2_live_slates.jsonl --scratch calibration/write_triage_eval_set_v2_rounds`.
`shards` re-renders the gitignored `shards/`, `texts.json` and
`rater_prompt.md` byte for byte from the committed key. `assemble` refuses
while any pair is not fully labelled.
`fused-memory/tests/test_write_triage_eval_set_committed.py` pins the
committed files against each other. It checks that a re-computation
reproduces the summary, the split and the cross-family report, and it holds
whatever the ratings say.

## 3. The offline population

There are 484 writes and 2,410 pairs (`summary.pairs.offline`). These are the
production-judged writes of the 658-write write-time draw of the 2026-10-05
frozen population
(`scripts/run_write_triage_population_arms.py::draw_run_set`, `slates=WRITE_TIME`):

- **Exclusions**: reconciliation markers and writes declaring attach keys,
  which production never triages.
- **Snapshot**: sha256 `55398cd73c9a355d05f9454a2e1cca9e7da760bd89df141b2f6b5b8f28293e38`
  (`summary.inputs.snapshot_sha256`, the same as
  `write_triage_population_write_time.json`).
- **Stratum**: `offline-write-time`, drawn from 1,102 judged writes
  (`summary.strata`).
- **Projects**: dark_factory 288 and reify 196, created 2026-09-29 to
  2026-10-05.
- **Slates**: each write's slate holds the frozen candidates created strictly
  before it, and its targets are the top 5 production's selector
  (`write_triage_judge.select_judge_candidates`) would show.
- **Texts**: as frozen on 2026-10-05.

## 4. The live population

There are 400 writes and 1,846 pairs (`summary.pairs.live`), rebuilt from the
2026-10-19 soak by `scripts/rebuild_write_triage_live_slates.py`. The
figures below come from the live-slates header.

- **Eligibility**: the soak journal's writes in the window from
  2026-10-11T13:35:00Z. Each must come from an agent or a full reconciliation,
  be Mem0-primary, have one memory id, and not be caller-owned (no caller
  attach, no declared attach keys). Eligible writes: dark_factory 754,
  reify 689, scoping_intake 49, solar_challenge 828, solar_challenge_platform
  572.
- **The slate and band**: production retrieval is re-run as it would have
  answered then. That means the 20 nearest by cosine among the project's
  unplanned Mem0-primary records, cross-category, with children hoisted to
  their canonical. The band is re-decided by `decide_band` at the soak edges
  t_high 0.8868282526657318 and t_low 0.5229672433064231. Only the judge band
  is kept. Judge-band populations (`summary.strata`): dark_factory 732,
  reify 666, scoping_intake 48, solar_challenge 760, solar_challenge_platform
  553.
- **Vectors and time**: the vectors are the soak's embedding space
  (text-embedding-3-small, 1536-d), read from the 2026-10-19 Qdrant dump
  `data/qdrant-dump-2026-10-19` (per-file sha256 in the header). A candidate
  counts only when it was created strictly before the write's
  `intent_created_at`. Production searched without a time filter, and this
  mask stands in for the records that did not exist yet. Each text is the one
  mem0's `history.db` gives for the write's instant.
- **Edited and deleted candidates**:
  - 35 slate candidates on 25 writes were edited after their write, and 9 of
    the 1,846 live targets are among them (`edited_after_write`).
  - Records deleted since the write are counted, not restored: 14,789 in
    total across the 400 writes, mostly 37 per write. history.db carries no
    project, so the count spans every project and is an upper bound on what
    any one slate lost.
- **Leavers and unrecoverable writes**: there are 2 leavers. These are
  production-judged attaches whose rebuilt band is restated, not judge. There
  are also 6 unrecoverable writes, whose own vector is missing from the dump.
  Both kinds are listed in the header and never replaced.
- **Comparison with team A**: the rebuilt band winner matches team A's
  10-19 rebuild on all 1,817 compared writes.
- **Allocation**: 400 writes are dealt round-robin over the projects in a
  seeded order per project (`draw_live_sample`, seed 7160), equal and capped
  at each project's population. That gives 88 each, and scoping_intake its
  whole 48. Production outcomes: amended 174, stored 150, contested 76.
  Strata are `live-<project>`.

## 5. The protocol

`write_triage_reader_metrics.md` §4 is applied to every top-5 pair on the
current basis (the text the judge was shown):

- **Rounds**: an Opus primary and an Opus second rate every pair blind. Lines
  that break the row contract or quote text not shown are re-asked. A third
  Opus rater, in rounds with a new id each, answers only the decisive ties and
  OPEN follow-ups. The round machinery is
  `scripts/write_triage_rating_rounds.py`, which the soak baseline builder
  shares.
- **Volume**: 9,367 rater lines (`summary.rating_volume`). Primary and second
  each rated 4,256 pairs. Third raters rated 264 distinct pairs in three
  rounds: third_00 (opus-third-00, 236 items), and third_01 (opus-third-01,
  38) and third_02 (opus-third-02, 1), both after the re-rate. No line ever
  needed a re-ask (`summary.reasked` 0).

**Why FULL_LABEL_OUTCOME is restated.** A pair is fully labelled when its
derivation under `build_write_triage_eval_set.py::FULL_LABEL_OUTCOME`
(restated) is not Incomplete. That derivation asks CONFLICT, then HOME on both
branches, then LOSS beside HOME YES and CONFLICT NO, plus the status-quo
stored reading. That is every decisive binary any judge outcome reads. A
stored item with earlier_ids would skip HOME on CONFLICT-YES pairs, which
amended, contested and restated attaches all read. Completeness, third-round
routing and the kappa consumed set are therefore per pair. The scorer's
per-write items cannot carry several pairs of one write, so kappa uses
`score_write_triage_reader.py::kappa_over_consumed` directly.

**Why no v1-derived rows.** A v1 seven-word verdict fixes at most one binary
(RELATED gives HOME NO; SAME/SUBSUMED gives CONFLICT NO). The raters would
have to answer the rest anyway. The soak's derivation check that held such
rows was measured on the soak's adjudications, not on λ's corpus, so it does
not transfer. Of the 2,410 offline pairs, 732 were already rated under v1 in
λ's `write_triage_pair_verdicts.jsonl`. They were rated again from scratch
here, and the other 1,678 were rated for the first time. λ's words are kept
as an independent cross-check on the RELATED boundary (§10). The v1 file is
untouched.

## 6. The brief revision: v2.1 to v2.2 (task 7703)

**Why.** Round 0, rated under brief v2.1 (commit `c7b682bcb9`), missed the
LOSS kappa floor: 0.598 on 257 consumed pairs, against HOME 0.8796 and
CONFLICT 0.8717. These figures come from the round-0 assembly before the
re-rate and are not in the JSON. §4 of the metrics doc forbids scoring
through a floor miss, so Leo ruled that the brief be refined, the LOSS pairs
re-rated, and the set finished.

**What v2.2 changes** (commit `169dcb2ef0`): only the LOSS question and
anchors 5 to 8. HOME, CONFLICT, SCOPE, action_changing, the output format and
the procedure are unchanged.

- **The test**: v2.1 asked whether someone acting on PARENT "would go wrong,
  or lack something they need". v2.2 puts each thing CHILD adds to one test: a
  reader acting on PARENT alone, in PARENT's situation, would act wrongly or be
  unable to finish without it.
- **What passes**: only three kinds of thing. A caveat or condition under
  which PARENT's claim does not hold. An unexpected consequence its reader
  would act on wrongly. A missing step, value or location PARENT's instruction
  needs, or a remedy for a hazard PARENT names.
- **What never passes on its own**: another route to PARENT's end; PARENT's
  rule applied or widened; a further sighting or measurement; provenance and
  backing; a side fact; confirmation. Anchors 5 to 8 illustrate these shapes.

**The re-rate.** The `rerate` subcommand
(`write_triage_rating_rounds.py::rerate_round`) keyed every pair whose
derivation then read LOSS. That was 290 pairs, re-rated as round 1 (shards
`rerate_01_primary_*` and `rerate_01_second_*`) by fresh blind Opus primary
and second raters under the v2.2 brief. The key and the summary record it as
`rerates: [{version v2.2, round 1, pairs 290, brief_sha256
7cd0bad3b1845c40d044b27946c703ba3d20a853aef6abd6140b5df8b56540c1}]`.

- **Supersession**: the round-1 lines supersede both raters' round-0 lines on
  those pairs.
- **Retirement**: the v2.1 third rater opus-third-00 is retired there, and
  its 54 lines on re-rated pairs are dropped from the verdicts. Third rounds
  third_01 and third_02 then resolved the ties the re-rate opened.
- **Pins**: the committed test pins all of this. It checks that the summary's
  `rerates` is what key.json implies, and that the brief sha256 is the
  committed brief's. It checks that every re-rated pair's primary and second
  lines come from round 1 or later, and that no retired third rater votes on
  a re-rated pair. It also checks that every pair whose derivation reads LOSS
  is a re-rated pair.
- **Result**: every LOSS answer the set consumes was given under v2.2. 282
  pairs read LOSS now, all among the 290; the other 8 re-rated pairs no
  longer do, because their v2.2 raters moved HOME or CONFLICT. A pair that
  never read LOSS keeps its v2.1 lines. Their HOME and CONFLICT wording is
  the same in v2.2, and nothing consumes their LOSS answers.

**The LOSS prevalence shift.** v2.2 counts only needed detail: a caveat, an
unexpected consequence, a missing step or a remedy. v2.1 also counted
detail that was useful but not needed. LOSS YES prevalence on the 290 pairs
fell from 257 / 290 = 0.886 under v2.1 (resolved round-0 lines) to 108 YES of
the 280 decided LOSS-reading pairs = 0.386 under v2.2 (2 UNCLEAR, 8 no longer
read). Both figures are counted from the committed rater files, not taken
from the JSON. This narrowing is deliberate.

- **Comparing across versions**: a figure that rests on LOSS (M1's hidden
  detail, the hidden_detail cell, duplicate_clutter's harm) is not comparable
  between a v2.1-rated corpus and a v2.2-rated one. The soak baseline of the
  metrics doc §8 is v2.1-rated. How far its LOSS-resting figures would move
  under v2.2 is unmeasured, but the direction is down. A comparison must re-rate
  one side's LOSS-consuming pairs under the other's brief first.
- **The score_reader label**: `score_write_triage_reader.py::score_reader`
  now takes the brief label as a required `rubric` keyword (CLI `--rubric`),
  and the report records it. A report over this set should say `v2.2`.

**The audit of the shift.** An auditor seat checked the v2.2 NO answers for
false NOs over the 290 re-rated pairs. This was its own analysis, not in the
JSON.

- **Transitions**: it compared primary/second LOSS cells, counting only pairs
  where both raters of a version gave HOME YES and CONFLICT NO. v2.1 YES/YES
  went to v2.2 NO/NO 103 times, stayed YES/YES 82 times, and went NO/YES 16
  and YES/NO 7 times. v2.1 NO/NO stayed NO/NO all 19 times, and all 21 v2.1
  splits went NO/NO. No v2.1 LOSS NO became YES.
- **The sample**: in a seeded sample of 60 of the 103 YES/YES to NO/NO flips
  (`random.Random(7160)`), the v2.2 raters applied v2.2 correctly in 60 of 60.
- **Harm**: judged on real harm to a reader acting on PARENT alone, the 60
  were 58 NO-HARM, 1 HARM and 1 UNSURE. The NO-HARM pairs are v2.1
  over-counting: another route 23, side fact 24, rule applied or widened 18,
  sighting 10, provenance 7, confirmation 3, completion record 2 (a pair can
  carry more than one).
- **The one HARM** is `cb72f021…/39c8cf90…`. Exclusion item 1 says "PARENT's
  own route is taken to work", and that hides a mode in which PARENT's route
  quietly does not apply: a planning_mode submit creates no ticket.
- **Residual risk**: the false-NO rate is about 2% (1 / 60), with a 95% upper
  bound of about 8.9%. That is at most about 9 of the 103 flips.
- **Verdict**: v2.2 is sound. The auditor recommends no wording change,
  because the only fix would reopen three splits the refiner had resolved.

## 7. Agreement

Cohen's kappa, primary against second effective lines, over the consumed
answers of decided pairs (`summary.kappa.by_binary`). The floor is 0.7 on at
least 60 consumed pairs, for home, conflict and loss.

| Binary | n consumed | Kappa | Floor met |
|---|---|---|---|
| home | 4254 | 0.9119 | yes |
| conflict | 4254 | 0.8684 | yes |
| loss | 270 | 0.803 | yes (brief v2.2) |
| scope | 0 | none | (not consumed by the restated derivation) |
| action_changing | 0 | none | (not consumed) |
| digest | 0 | none | (computed, not consumed) |

Every floor binary meets the floor, so the set meets the metrics doc §4's
condition for scoring. LOSS has 270 consumed pairs, not the 280 decided pairs
that read it. On the other 10, the primary and second did not both answer
LOSS (counted from the verdicts, not in the JSON).

## 8. Completeness

There are 4,256 pairs. 4,254 are decided and 2 are unclear (`summary.labels`);
none is incomplete. The two unclear pairs (`summary.unclear_pairs`) are both
LOSS, where both v2.2 raters answered UNCLEAR on a truncation:
`0e6bd429-4b0a-423b-99d5-e0cae764b108` / `6fb4a744-7cf3-4ced-8907-240710aadf27`
and `7e9aac4e-64ee-43cb-a797-6dedc8fe96b5` / `e0a41fcd-0745-426b-bf71-45466e6d34e0`.

## 9. The split

**The rule** (`split.rule`, `build_write_triage_eval_set.py::SPLIT_RULE`):

- **Test**: every live write of solar_challenge_platform, and every other
  live write created at or after the UTC midnight inside the live window
  whose test split is closest to 300 writes. A tie goes to the later
  midnight.
- **Tuning**: every other write, ordered by `sha256('7160:<entry_id>')`. Dev
  is the first half, rounded down, and train the rest.
- **Label-blind**: verdicts never decide membership. They enter only
  `test_sha256` and `verdicts_sha256`.

**Why this project and this size.** This follows seat C's design: later
dates, plus one solar project, about 300 writes, hash-pinned.

- solar_challenge_platform is held out whole, so the test split includes a
  project no tuning row comes from. It is the smaller of the two solar
  projects (572 eligible against 828 here), so the larger one stays
  available for tuning.
- The time cut holds out the latest live writes, so a variant tuned on
  earlier traffic is tested on later traffic.
- The 327 test writes carry 1,491 pairs (counted from the committed files).
- No offline write is in test. The offline draw is older and from two
  projects only.
- The rule reads only ids, sources, projects and dates, so no label chooses
  the split. Freezing it before any tuning run exists makes the hash a real
  no-peek guard.

**The frozen split.** Cutoff `2026-10-14T00:00:00+00:00`.

| Split | Writes | By source and project |
|---|---|---|
| test | 327 | live: dark_factory 57, reify 57, scoping_intake 48, solar_challenge 77, solar_challenge_platform 88 |
| dev | 278 | live: dark_factory 14, reify 16, solar_challenge 4; offline: dark_factory 149, reify 95 |
| train | 279 | live: dark_factory 17, reify 15, solar_challenge 7; offline: dark_factory 139, reify 101 |

Hashes, each the sha256 of the committed JSONL form (one sort_keys object per
line):

- `test_sha256` `3ca40b80af8db3e239346445a73b1f99100f60a2604272ab43e5719efdd467bf`:
  the test writes, then their verdicts.
- `writes_sha256` `059a1bf12ef812179d0730dde01c55a1e3f058a116bb48d218c54bc2f8036596`.
- `verdicts_sha256` `695e9dc39b4934996896b9ccbf1b970027081f4bc87fd9fddfa09b3c68523a2e`.

**Verify** with `build_write_triage_eval_set.py::verify_split(split, writes,
verdicts)`. It names any write that moved, appeared or vanished, and refuses
when `test_sha256` drifts. Tuning verdicts may be refined; test verdicts may
not.

**Using it.** No tuning run may read test rows. A consumer takes its rows
through `build_write_triage_eval_set.py::tuning_writes`, which never yields a
test write. It scores the test split at most three times over the life of a
variant search (seat C). The split was frozen before any tuning run existed,
and nothing in task 7160 read test-split figures to choose anything.

## 10. The cross-family re-rate

**Rater**: gpt-6.1-sol at effort high, through the Responses API. This is a
different family from the Opus raters, and not the judge's configuration (sol
at effort low). Each pair went alone: the answer contract, with the brief as
it read on 2026-10-29 (v2.2, sha256 `7cd0bad3…0c1`), plus the pair rendered
exactly as the Opus raters saw it.

**Selection**:

- every decided pair whose primary and second split on any consumed answer:
  220 discordant;
- a seeded control of 100 concordant pairs (`control_pairs`, seed 7160).

All 320 were rated in one call each, with no failures and no re-asks
(`write_triage_eval_set_v2_cross_family.json`, `coverage`).

**Spend**: 4.05 USD at list price (`spend_usd` 4.05048). The task's estimate
was about 1 to 3 USD; the run went past it because a call cost about 0.0127
USD and the selection was 320 calls. It stayed under the 4.50 USD cap the
lead approved for the run, which was inside the plan's 15 USD cap. A 3.00 USD
leg stopped with 2 discordant pairs and every control unrated, and was
resumed after approval.

**Agreement with the resolved Opus label**:

| Selection | Binary | n | Agree | Rate | Kappa |
|---|---|---|---|---|---|
| discordant | home | 220 | 173 | 0.7864 | 0.5666 |
| discordant | conflict | 219 | 150 | 0.6849 | 0.383 |
| discordant | scope | 84 | 68 | 0.8095 | 0.6285 |
| discordant | action_changing | 82 | 73 | 0.8902 | 0.6276 |
| discordant | loss | 46 | 30 | 0.6522 | 0.3622 |
| control | home | 100 | 98 | 0.98 | 0.8643 |
| control | conflict | 100 | 95 | 0.95 | 0.7346 |
| control | scope | 8 | 8 | 1.0 | 1.0 |
| control | action_changing | 6 | 5 | 0.8333 | 0.0 |
| control | loss | 6 | 5 | 0.8333 | 0.0 |

The rendered `.md` gives the same against each Opus rater, with Wilson 95%
intervals.

On the control, where both Opus raters agree, the cross-family rater agrees
with them on 98 of 100 HOME and 95 of 100 CONFLICT answers. Where the Opus
raters split, it agrees with the resolved label well below that: HOME 0.7864,
CONFLICT 0.6849, LOSS 0.6522. Those pairs are hard for a third family too, so
their resolved labels are the set's least certain.

On LOSS discordants it agrees with the primary 25 / 34 and with the second
only 11 / 32. The control holds only 6 LOSS answers, too few to read.

**The RELATED/HOME boundary** (`related_boundary`):

- On the 69 pairs whose Opus raters split on HOME, the cross-family rater
  says HOME YES on 58 and agrees with the resolved label on 36. Where Opus is
  divided on whether CHILD's central claim is PARENT's, sol mostly says it
  is, so the resolved Opus HOME label sits on the stricter side of the
  boundary.
- Against λ's v1 words, cross-family HOME YES / NO:

| v1 word | YES | NO |
|---|---|---|
| RELATED/UNRELATED | 1 | 18 |
| SAME/SUBSUMED | 6 | 0 |
| EXTENDS | 33 | 13 |
| CORRECTS | 13 | 2 |

The cross-family rater keeps λ's RELATED pairs out of HOME (18 / 19) and
SAME/SUBSUMED in (6 / 6). The boundary in dispute is EXTENDS, which it puts
in HOME 33 times of 46. That is the case R7 retired as a v1 row
(extends → HOME YES). On the soak, the Opus raters agreed with it on only
0.430 of compared answers (metrics doc §8, derivation check).

**What this says about Opus rater bias.** There is no sign of a bias shared
across the whole corpus: on concordant pairs a different family agrees
almost entirely. The disagreement is concentrated where the two Opus raters
already split. There, the Opus resolution leans HOME NO against sol's YES,
and CONFLICT kappa is low (0.383). So the set's uncertainty sits on the 220
discordant pairs, about 5% of the 4,256, and mostly on HOME's boundary with
"related" and on CONFLICT.

## 11. How to score a judge variant

1. Run the variant on each tuning write's slate (`tuning_writes`), and record
   its outcome and target.
2. Build one `score_write_triage_reader.py::ReaderItem` row per write:
   - **An attach outcome** (restated, amended or contested) is read against
     its one target. Its `target_id` is the chosen target, which must be one
     of the write's targets.
   - **A stored write** names no target. Its `earlier_ids` are all of the
     write's targets.
   - **The other fields**: `band` is `judge`, `project` and `stratum` come from
     the write row, and the unknown flags are None.
3. Score with `score_reader(items, verdicts, rubric='v2.2', strata=...)` (or
   the CLI with `--rubric v2.2`) over `write_triage_eval_set_v2_verdicts.jsonl`.
   Every binary any outcome's derivation reads is already labelled on every
   pair (§5), so an item built this way needs no new rating.

Weight by `summary.strata` to read population rates. Offline and live
strata stand for different populations: the 2026-10-05 freeze and the soak's
judge band.

## 12. Caveats

- The offline writes are only dark_factory and reify, frozen on 2026-10-05.
  The live writes are the soak's five projects. Neither holds a write from
  after 2026-10-19.
- The live slates are reconstructions under a created-before mask on a
  10-19 dump. Records deleted before the dump are missing from the slates.
  The header counts them (§4) but cannot restore them.
- LOSS rests on brief v2.2, and on the soak baseline v2.1 (§6). Figures that
  read LOSS do not compare across the two without a re-rate.
- The auditor's residual false-NO estimate for v2.2 LOSS is about 2%, with an
  upper bound near 8.9%, on one mode: a CHILD scoped to a variant in which
  PARENT's own route quietly does not apply (§6).
- SCOPE and action_changing are never consumed by the restated derivation, so
  this set carries no kappa for them. Their rater answers are in the verdicts
  where CONFLICT is YES.
- The discordant pairs, about 5%, are the set's least certain labels. A
  third model family agrees with their resolved labels at 0.65 to 0.79 by
  binary (§10).
- Two pairs are UNCLEAR on LOSS and count in no rate (§8).
