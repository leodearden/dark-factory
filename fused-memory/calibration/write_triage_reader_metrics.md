# Write-triage reader metrics (rubric v2) and the 3131 post-fix measurement

This document defines the metrics that judge write triage by what a later
reader of the corpus meets, and the procedure that measures them for task
3131's release. Raters answer the questions in
`write_triage_rater_brief_v2.md`. `fused-memory/scripts/write_triage_reader_rubric.py`
turns their answers into cells, and `fused-memory/scripts/score_write_triage_reader.py`
turns cells into the figures below. The scorer reports and never gates: every
gate on this page is applied by a person reading its report.

Design source: the seat B report and the decision memo,
`/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/correctness-analysis-2026-10-19.md`
§2 and §6 (gitignored, main checkout). Rulings 5a–5d and 5f are Leo's, of
2026-10-19, recorded on task 7157.

## 1. Why readers, not pairs

The v1 scale (`write_triage_rater_brief.md`, scored by
`fused-memory/scripts/score_write_triage_pairs.py`) asks whether a write
belongs under the record it was attached to. Harm, though, lands on whoever
later reads that record, and attached writes are shown to them very
unevenly:

- An amendment reaches a reader of its parent as a digest of its first 240
  characters (`fused-memory/src/fused_memory/server/grouped_read.py::_DIGEST_CHARS`),
  and at most 10 digests are listed (`::_AMENDMENT_DIGEST_CAP`).
- A sighting (a restated write) appears only as a count, and briefings do not
  render that count: `orchestrator/src/orchestrator/agents/memory_recall.py::GROUPED_CHILD_KEYS`
  carries amendments and matched children, not `sighting_count`.
- A contested child is never digested. It stays a top-level result, tagged as
  contesting its parent (`grouped_read.py::is_contested_child`).

So the same pair can be harmless as an amendment and harmful as a sighting.
Rubric v2 asks raters facts about the two texts (HOME, LOSS, CONFLICT, SCOPE,
DIGEST) and derives correctness from the outcome the write actually got.

## 2. Cells

A cell is one way an outcome leaves a reader worse off, and a write may fall
in several. The one definition of the eight cells and of how answers map to
them is `write_triage_reader_rubric.py::derive`; their rules, worded as the
report carries them, are `write_triage_reader_rubric.py::CELL_RULES`. Every
write is derived twice: under triage, with the outcome it got, and under the
status quo, where the same write is stored flat and read against the record
it would otherwise have joined.

## 3. Metrics

Every proportion is reported as n, hits, rate and a Wilson 95% interval over
the decided sample. With a strata map it also carries a population rate, each
item weighted N_h / n_h, and a stratified bootstrap 95% interval. The bootstrap
resamples each stratum's whole sample, so a metric over a sub-population (M1,
M3, M4's splits) also carries the uncertainty in how many sampled writes fall
in it. An item whose deciding binary resolved UNCLEAR is left out of n and
counted as `unclear`.

An item's regimes stand or fall together. It is decided only when its triage
cells, its status-quo cells and, in the M1 band, M1 are all decided. One
UNCLEAR deciding binary in any of them leaves the item out of every metric,
even a metric that reads only another regime. For example, take a contested
write with CONFLICT NO and HOME YES whose LOSS is UNCLEAR. Only the status quo
asks about LOSS, yet the write is left out of M2, M4 and M5 too. In the same
way, a gap in any one regime makes the scorer refuse the corpus, because harm
compares the two regimes. `population.unclear_entry_ids` names every item
left out.

| Metric | Rule | Population and unit | Sample | Status |
|---|---|---|---|---|
| **M1** hidden information in the high-cosine band | the write's dispute or needed detail is not visible to a reader of the parent: CONFLICT, or HOME with LOSS for a restated write, or HOME with LOSS and DIGEST=NO for an amended one (`derive`, `Decided.m1_hidden`); contested and stored writes are visible | every write whose band-winner cosine was >= t_high when it was written (about 105 a week), whatever its outcome; unit = write | census | **gate: point estimate <= 0.15**, Wilson reported. Soak baseline 0.50 (v1 proxy) |
| **M2** ack truth per outcome | the outcome triage gave is correct, meaning no triage cell | attach and stored acks; unit = write, per outcome | 100 per outcome, stratified outcome × project | report-only |
| **M3** hidden corrections | of the writes that dispute their parent or an earlier record (CONFLICT), the share that ended restated, amended, or stored with no mark on the record they dispute | every CONFLICT write; unit = write | >= 60 CONFLICT writes, drawn from the contested census, the M2 samples, and 150 stored writes rated against their nearest earlier records | **gate: bound unset — Leo sets it.** The critic's soak-equivalent after F3/F4 is >= ~0.06–0.07 |
| **M4** wrong home | wrong_home over decided attaches, split direct vs via-sibling and same vs cross category | attaches; unit = write | shared with M2 | reported against 0.20 |
| **M5** conflict-flag precision | contested writes with CONFLICT, split by SCOPE (central, secondary, unknown) | contested writes; unit = write | census or 100 | report-only |
| **M6** read-time drift | children whose triage cells differ when re-rated against the parent's text at attach time | children whose parent was edited after they attached; unit = write | census | report-only until the edit guard of ruling 5d lands |

## 4. Rater protocol

- Two blind raters per item, given the brief verbatim, blind to outcome,
  system and key files. Raters answer every binary; the brief says when SCOPE
  and DIGEST apply.
- Agreement is Cohen's kappa per binary between the primary and the second
  rater, over the pairs both answered. A binary is good enough to score when
  kappa >= `score_write_triage_reader.py::KAPPA_FLOOR` on at least
  `::KAPPA_MIN_OVERLAP` overlap items. Below the floor, refine the brief and
  re-rate before scoring; never score through it.
- A third rater is asked only about decisive disagreements: a binary that is
  tied and that the derivation needs. The scorer names them, refusing with
  `score_write_triage_reader.py::IncompleteJudgementError` until each is
  resolved. A tie on a binary the derivation never asks for is left as it is.
- UNCLEAR only where a truncation falls at the deciding point, with the point
  named. Such items are left out of every rate and counted.
- Blind anchors, items whose answers are already known, are mixed into each
  primary shard to detect drift between batches;
  `build_write_triage_v2_soak_baseline.py::ANCHORS_PER_SHARD` sets how many.

## 5. Reconstructing write-time state

Each item needs the state at the moment the write was triaged, not the state
when the measurement is run.

- **The write**: its text and metadata come from the write journal
  (`data/reconciliation/write_journal.db`, table `mem0_intents`) or from
  Qdrant.
- **Outcome**: from the child's metadata: `kind` (sighting or amendment),
  `parent_id`, and `x_contested`. A write with none of them was stored.
- **Band**: the cosine to the band winner against the t_low and t_high in
  force at the time, read from the config history
  (`fused-memory/config/config.yaml`, `write_triage.t_high` / `t_low`), not
  today's values.
- **Hoist** (`via_sibling`): the write's cosine to its parent compared with
  its cosine to the parent's earlier children; it is via-sibling when an
  earlier sibling is closer than the parent.
- **The parent at attach time**: from mem0's `~/.mem0/history.db`, table
  `history`. Take the `old_memory` of the earliest UPDATE whose `updated_at`
  is after the child's `created_at`; with no such UPDATE the current text is
  the attach-time text. The write journal's `update_memory` parameters are cut
  at 200 characters and cannot be used. Open the database read-only
  (`file:<path>?mode=ro`) with `PRAGMA temp_store=MEMORY`, or sorted queries
  fail with "unable to open database file".
- **A stored write's earlier records**: the top 5 by cosine among records
  strictly earlier than the write. An undated record is never earlier; this is
  the rule of `score_write_triage_pairs.py::WriteHomes`.
- **Supersession** (ruling 5c): `supersedes_target` is true when the parent is
  listed in the child's reserved `supersedes` metadata key, and
  `superseded_marks` lists the earlier records carrying a superseded-by mark
  for the write.

## 6. Harm weights — PROVISIONAL until Leo ratifies them (ruling 5f)

`score_write_triage_reader.py::PROVISIONAL_HARM_WEIGHTS` prices each cell. The
report labels them `provisional`, and any other map is labelled `custom`.

| Cell | Weight | Reader consequence | Continuity with D20 |
|---|---|---|---|
| hidden_conflict | 3 | a dispute no reader of the record ever sees | the top weight, as v1's contested misfile; above orphaned_conflict because hidden is worse than flat and visible |
| digest_conflict | 2 | a dispute shown only as an untagged 240-character digest | between hidden (3) and flat (1.5) |
| false_conflict | 2 | a sound record shown as in doubt | = v1 false_contested 2 |
| orphaned_conflict | 1.5 | a flat dispute that a search can find but that is never linked to the record | below v1 contradiction_unflagged (2), which priced a hidden and a visible correction the same |
| hidden_detail | 1 | a needed fact reduced to a count | new: D20 had no cell for it; the status quo charges nothing, because a flat write is visible |
| wrong_home | 1 | framing harm; a matched child stays pinned under the wrong parent | = v1 misfile_attach 1; false_conflict + wrong_home = 3 = v1 contested misfile |
| attached_supersession | 1 | a replacing claim filed under the record it replaces (5c) | new, priced as a wrong placement |
| duplicate_clutter | 0.5 | a redundant flat record | = v1 placeable_left_flat 0.5 |

Ruling 5b removes true secondary disputes from false_conflict, which answers
B's complaint that D20 weighted them 3, without a special case.

The frame stays comparative: harm per write under triage against harm per
write under the status quo (every write stored flat), on the same writes. The
report gives the per-write delta with a seeded stratified bootstrap 95%
interval (`::BOOTSTRAP_RESAMPLES`, `::BOOTSTRAP_SEED`). To score with another
map, pass `--harm-weights PATH` (a JSON `{cell: weight}`; a cell left out
weighs 0) or `score_reader(harm_weights=...)`.

## 7. The 3131 post-fix measurement

**Preconditions.** Deployed: F1 with the server-side completion backstop, F3′
(the restate band goes through the judge, which may only say `restates` when
nothing is new) and F4 (supersessions stored standalone, with the replaced
record marked). The window opens when the last of them is live, runs at least
7 days, and is extended until M3 has its >= 60 CONFLICT writes.

**Steps.**

1. Extract every triaged write in the window, with its state reconstructed as
   in §5.
2. Sample: the M1 census of band writes; 100 per outcome for M2 and M4,
   stratified by project; the contested census; 150 stored writes against
   their nearest earlier records for M3; and the M6 census of children whose
   parent was edited after attach. Record each stratum's population in a
   strata file.
3. Build shards as `fused-memory/scripts/build_write_triage_v2_soak_baseline.py`
   does: the brief verbatim, texts capped at 4,000 characters, the digest
   rendered by production's `grouped_read.py::_digest`, opaque item ids and a
   separate key.
4. Rate per §4: two blind raters on every item, then a kappa check.
5. Score with `score_write_triage_reader.py --strata`. For each gap a refusal
   names, ask a third rater, then re-score until the scorer accepts the corpus.
6. Read the release condition off the report and the window's logs.

**Release condition** (memo §6, decision 8a). All of the following must hold:

- M1 at or below 0.15 (point estimate; Wilson reported).
- M3 within the bound Leo sets.
- Zero cross-task completion attaches: no completion record attached under
  another task's completion record.
- The fail-open rate and judge latency are no worse than the soak's: 0.37%
  fail-open, and a judge p95 bracketed at 6.1–13.0 s (unambiguous subset
  6.34 s; `volume-health/judge_latency_bracket.json` in the soak bundle).

**Report-only:** M2 per outcome, M4 against 0.20, M5, M6, and harm per write
under the provisional weights.

**After the embedder cutover (6807)** re-run the whole measurement: the band
edges and every cosine-derived threshold move with the embedder.

## 8. Soak baseline

The 2026-10-19 soak's 553 adjudicated attaches, re-based on rubric v2. The
report is `write_triage_v2_soak_baseline.json` (rendered as
`write_triage_v2_soak_baseline.md`). It is the scorer's output over
`write_triage_v2_soak_{items.jsonl,verdicts.jsonl,strata.json}`, and
`fused-memory/tests/test_score_write_triage_reader.py::TestTheCommittedSoakBaseline`
pins that re-scoring those inputs reproduces it. Every figure below is quoted
from it unless marked otherwise.

**This baseline is scored through a kappa floor miss and through a failed
derivation check (both below).** §4 forbids that for the 3131 measurement.
Read the baseline as the record of what was measured, not as a calibrated
comparison point.

### Provenance

- **Inputs**, read-only: the soak bundle's `correctness/adjudications.jsonl`,
  `items_all.json` (current parent text, read from Qdrant on 2026-10-19),
  `hoist.json` and `attaches_soak.json`, and `~/.mem0/history.db` for
  attach-time text. Built by `fused-memory/scripts/build_write_triage_v2_soak_baseline.py`
  with seed 7157. Shards, the key and the raw rater output are in a gitignored
  scratch directory, `data/write-triage-v2-rebase-2026-10-19/`.
- **What the v1 labels fix**: `build_write_triage_v2_soak_baseline.py::v1_determined`.
  312 items needed rating, and 241 were fully fixed by v1.
  `write_triage_v2_soak_derivation.json` counts these 312 as
  `rated_current_pairs`. Its `anchor_pairs` (35) and `attach_time_pairs` (13)
  are the anchors and attach-time re-reads below.
- **Rating volume**: 459 blind ratings in 11 shards. Opus primary: 7 shards,
  347 items, of which 35 are fully fixed anchors. Opus attach-time: 1 shard, all
  13 parent-edited items, every one recoverable from history.db. Sonnet second:
  2 shards, an 80-item seeded overlap. Opus third: 1 shard, 19 decisive ties.
  These raters read a brief that said "about 10" anchors per shard, while the
  builder mixed in 5. The brief now says "a few", and no other wording has
  changed.
  16 binaries remain tied and are not decisive: loss 8, digest 5, scope 3.
  One item resolved UNCLEAR and is left out of every rate.
- **Kappa, primary against second.** In the corpus, a rated answer is kept only
  on a binary that v1 left open. On that basis no binary meets the floor:
  home 0.00 (n 3), loss 0.40 (n 80), conflict 0.58 (n 74), scope 0.49 (n 18),
  digest −0.04 (n 48). Over every answer both raters gave, in the scratch
  ratings: home 0.95 (n 80), loss 0.40 (n 80), conflict 0.62 (n 80), scope 0.59
  (n 21), digest −0.04 (n 48). Raw agreement on digest is 42 of 48, but almost
  every answer is NO, so kappa is near zero. LOSS, CONFLICT and DIGEST need the
  brief refined before the 3131 measurement (§4).
- **Derivation check** (`write_triage_v2_soak_derivation.json`): rated answers
  compared with the v1-fixed answer on the same binary. Overall: home 261 of
  389, conflict 50 of 66, scope 30 of 38, loss 2 of 2. Per v1 word:
  - RELATED/UNRELATED → HOME NO: 223 of 223 hold.
  - SAME → YES: 6 of 6 hold. SUBSUMED → YES: 2 of 3 hold.
  - **EXTENDS → HOME YES: 30 of 157 hold.** Raters, who agree with each other
    on HOME at kappa 0.95, read most v1 EXTENDS pairs as a different claim.
  - CORRECTS + action-changing → CONFLICT YES: 38 of 40 hold.
  - SAME/SUBSUMED/UNRELATED → CONFLICT NO: 10 of 10 hold.
  - **CORRECTS, not action-changing → CONFLICT NO: 2 of 16 hold.**

### Headline figures

Sample rate (Wilson 95%), then population rate (stratified bootstrap 95%).

| Figure | Soak baseline | Reference |
|---|---|---|
| M1 hidden in band | 18 / 39 = 0.46 [0.32, 0.61]; pop. 0.47 [0.32, 0.63] | v1 proxy 0.50; gate 0.15 |
| M2 restated correct | 13 / 40 = 0.33 [0.20, 0.48] | |
| M2 amended correct | 25 / 40 = 0.63 [0.47, 0.76] | |
| M2 contested correct | 315 / 472 = 0.67 [0.62, 0.71] | |
| M3 hidden corrections | 15 / 340 = 0.04 [0.03, 0.07]; pop. 0.36 [0.17, 0.50] | bound unset |
| M4 wrong home | 127 / 552 = 0.23 [0.20, 0.27]; pop. 0.30 [0.20, 0.40] | 0.20 |
| M4 direct / via sibling | 0.13 (n 426) / 0.56 (n 122) | |
| M5 conflict-flag precision | 325 / 472 = 0.69 [0.65, 0.73]; central 166, secondary 156, unknown 3 | |
| M6 drift | 9 of 13 parent-edited children change cells at attach time | report-only |
| Harm per write, sample | triage 0.87, status quo 0.94, delta −0.07 [−0.22, 0.08] | provisional weights |
| Harm per write, population | triage 0.70, status quo 0.46, delta +0.24 [0.12, 0.38] | provisional weights |

M3's ending split: restated 10, amended 5, contested 325. The sample rate is
dominated by the contested census. The population rate is dominated by the 5
amended conflicts, each of which stands for 30.6 amended writes.

**Sensitivity to the failed derivation rows** (scratch ratings, not committed,
and not reproducible from the committed corpus). This takes the raters' HOME on
all 114 v1 EXTENDS items in place of v1's YES:

- M2 amended falls to 0.25 (10 / 40).
- M4 rises to 0.34 sample and 0.61 population.
- M1 falls to 0.41.
- The population harm delta rises to +0.56 [0.42, 0.68].

Also taking the raters' CONFLICT on the 16 CORRECTS, not action-changing,
items moves M5 to 0.71 and M4 to 0.32 sample (0.61 population). Read the
as-built M4 as a lower bound on wrong homes, and M2 amended as an upper bound
on amended correctness.

### Caveats

- The soak holds no stored items, so stored cells and M3's stored endings are
  unmeasured.
- The status quo's earlier records are only each write's parent, so its
  orphaned_conflict and duplicate_clutter counts are a lower bound on its harm.
- Restated and amended are 40-item samples of 125 and 1,224, weighted by
  strata. The contested stratum is the census of 473.
- One contested item (`e7f06b8d-5bf4-4070-b085-e2db2d29d655`) resolved UNCLEAR
  and is left out.
