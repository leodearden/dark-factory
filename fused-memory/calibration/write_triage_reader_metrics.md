# Write-triage reader metrics (rubric v2.1) and the 3131 post-fix measurement

This document defines the metrics that judge write triage by what a later
reader of the corpus meets, and the procedure that measures them for task
3131's release. Raters answer the questions in
`write_triage_rater_brief_v2.md` (rater brief v2.2 since task 7160; v2.1 is commit
`c7b682bcb9`). `fused-memory/scripts/write_triage_reader_rubric.py`
turns their answers into cells, and `fused-memory/scripts/score_write_triage_reader.py`
turns cells into the figures below. The scorer reports and never gates: every
gate on this page is applied by a person reading its report.

Design source: the seat B report and the decision memo,
`/home/leo/src/dark-factory/data/write-triage-soak-2026-10-19/correctness-analysis-2026-10-19.md`
§2 and §6 (gitignored, main checkout). Rulings 5a–5d and 5f are Leo's, of
2026-10-19, recorded on task 7157. His rulings R1–R8 of 2026-10-21, recorded
on task 7174, revise the brief and the rater protocol as rubric v2.1.

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
Rubric v2.1 asks raters facts about the two texts (HOME, CONFLICT, LOSS, SCOPE,
action_changing), backed by verbatim quotes, and derives correctness from the
outcome the write actually got. DIGEST, whether what LOSS points to is visible
in an amendment's digest, is computed from the raters' loss spans
(`fused-memory/scripts/write_triage_reader_rubric.py::digest_visible`), not
asked.

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
| **M1** hidden information in the high-cosine band | the write's dispute or needed detail is not visible to a reader of the parent: CONFLICT, or HOME with LOSS for a restated write, or HOME with LOSS and DIGEST=NO for an amended one (`derive`, `Decided.m1_hidden`); contested and stored writes are visible. DIGEST is YES only when every LOSS-YES rater's spans all lie in the digest (`digest_visible`, resolved all-visible) | every write whose band-winner cosine was >= t_high when it was written (about 105 a week), whatever its outcome; unit = write | census | **gate: point estimate <= 0.15**, Wilson reported. Soak baseline 0.3736 under rubric v2.1 (§8); the v1 proxy was 0.50 |
| **M1 as amended** (counterfactual) | M1's rule with every band attach read as if it were amended: CONFLICT, or HOME with LOSS and DIGEST=NO (`Decided.m1_hidden_as_amended`) | M1's band writes that were attached; unit = write | shared with M1 | report-only |
| **M2** ack truth per outcome | the outcome triage gave is correct, meaning no triage cell | attach and stored acks; unit = write, per outcome | 100 per outcome, stratified outcome × project | report-only |
| **M3** hidden corrections | of the writes that dispute their parent or an earlier record (CONFLICT), the share that ended restated, amended, or stored with no mark on the record they dispute | every CONFLICT write; unit = write | >= 60 CONFLICT writes, drawn from the contested census, the M2 samples, and 150 stored writes rated against their nearest earlier records | **gate: bound unset — Leo sets it.** The critic's soak-equivalent after F3/F4 is >= ~0.06–0.07 |
| **M4** wrong home | wrong_home over decided attaches, split direct vs via-sibling and same vs cross category | attaches; unit = write | shared with M2 | reported against 0.20 |
| **M5** conflict-flag precision | contested writes with CONFLICT, split by SCOPE (central, secondary, unknown) and by action_changing (yes, no, unknown) | contested writes; unit = write | census or 100 | report-only |
| **M6** read-time drift | children whose triage cells differ when re-rated against the parent's text at attach time | children whose parent was edited after they attached; unit = write | census | report-only until the edit guard of ruling 5d lands |

## 4. Rater protocol

- Two blind raters per item, Opus primary and Opus second, on every item and
  every basis. Both get the brief verbatim and are blind to outcome, system
  and key files. Raters answer HOME and CONFLICT, plus the follow-ups the brief
  conditions on them; DIGEST is computed, never asked.
- Every rater line is checked before scoring. It must keep the rubric's row
  contract (`fused-memory/scripts/write_triage_reader_rubric.py::resolve_judgements`):
  the answer words, the follow-ups its own answers ask, and the evidence an
  answer needs. Every `conflict_quote` side and every loss span must be found:
  a whitespace-normalised substring of the text its rater saw (`::quote_found`).
  A rater whose line breaks the contract, or whose quote or span is not found,
  is asked the item again (the `reask` subcommand of
  `fused-memory/scripts/build_write_triage_v2_soak_baseline.py`, via
  `fused-memory/scripts/write_triage_rating_rounds.py::rating_problems`), and
  the later round's line supersedes the earlier one.
  `third` and `assemble` refuse while any such line is left.
- Agreement is Cohen's kappa per binary between the primary and the second
  rater, over consumed answers only. An answer is consumed when both raters
  gave it on a (record, binary) the derivation read
  (`fused-memory/scripts/write_triage_reader_rubric.py::Decided`, its
  `consumed` set), or on the SCOPE and action_changing that M5 splits a
  contested conflict by. An answer nothing reads, such as LOSS on a pair
  whose HOME resolved NO, is noise and is left out.
- A binary of `fused-memory/scripts/score_write_triage_reader.py::KAPPA_FLOOR_BINARIES`
  (home, conflict, loss) is good enough to score when its kappa is at least
  `::KAPPA_FLOOR` on at least `::KAPPA_MIN_PAIRS` consumed pairs. Where a
  binary has fewer consumed pairs, over-sample a stratum in which it is
  consumed until it has them. Below the floor, refine the brief and re-rate
  before scoring; never score through it.
- SCOPE below the floor on its own is recorded and reported with
  `escalate_info`, and does not block scoring. Leo ratified this reading of
  R5 on 2026-10-21 (task 7174, metadata `x_ruling_ratified`).
- A third rater, also Opus, is asked only about decisive gaps that the
  derivation needs: a tied binary, or an OPEN follow-up. A follow-up is OPEN
  when the raters' answers resolve every one of its preconditions, though no
  rater's own answers met them all, so no rater gave it
  (`fused-memory/scripts/write_triage_reader_rubric.py::Resolved`, OPEN). For
  example, a primary answering HOME YES / CONFLICT UNCLEAR beside a second
  answering HOME UNCLEAR / CONFLICT NO resolves HOME YES and CONFLICT NO with
  no LOSS. A third rater answers the whole item blind: one whose own HOME is
  YES and CONFLICT NO must give LOSS, and any other answer moves the
  preconditions, so the rounds go on. No rater is ever asked LOSS
  conditionally. The scorer names every gap, refusing with
  `fused-memory/scripts/score_write_triage_reader.py::IncompleteJudgementError`
  until each is resolved. Third raters come in rounds, each round a new rater
  id, because a resolved CONFLICT can open a tied LOSS. A tie on a binary the
  derivation never asks for is left as it is.
- UNCLEAR only where a truncation falls at the deciding point, with the point
  named, or where no rater reached a follow-up because a held v1 row fixed an
  answer it is conditioned on (`fused-memory/scripts/write_triage_reader_rubric.py::Judgement`,
  its `unreached_follow_ups`). A follow-up the raters' own answers leave open
  is not UNCLEAR: it goes to a third round. An item whose deciding binary is
  UNCLEAR is left out of every rate and counted.
- The brief is v2.2 from task 7160 (task 7703, commit `169dcb2ef0`). Only the
  LOSS question and anchors 5 to 8 changed from v2.1, whose text is commit
  `c7b682bcb9`. When a brief revision changes a question, every pair whose
  derivation reads that question is re-rated by fresh primary and second
  raters, and earlier third votes on those pairs are retired
  (`fused-memory/scripts/write_triage_rating_rounds.py::rerate_round`).
  `write_triage_eval_set_v2.md` §6 records why v2.2 was written and what it
  moved. Verdict rows do not record the brief, so the scorer takes the label
  from its caller (`score_write_triage_reader.py --rubric`).

## 5. Reconstructing write-time state

Each item needs the state at the moment the write was triaged, not the state
when the measurement is run.

- **The write**: its text and metadata come from the write journal
  (`data/reconciliation/write_journal.db`, table `mem0_intents`) or from
  Qdrant. Its memory id is in the same journal: the `memory_ids` of its write
  op's `write_ops.result_summary`. Open the journal read-only.
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
3. Build shards with `fused-memory/scripts/write_triage_rating_rounds.py`, as
   the soak builder does: the brief verbatim, texts capped at 4,000 characters, no digest,
   opaque item ids, a separate key, and a record of the texts each rater saw.
4. Rate per §4: two blind Opus raters on every item. Validate every line, and
   re-ask each rater whose line breaks the row contract or whose evidence is
   not found until none is left. Then check kappa over consumed answers.
5. Score with `score_write_triage_reader.py --strata --rubric <brief label>`. For the gaps a refusal
   names, run a third round with a new rater, and repeat until the scorer
   accepts the corpus.
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

**Which brief: open, for Leo.** This release condition was ratified under
brief v2.1. M1 <= 0.15, and M3 population <= 0.35 (esc-7301-1, recorded on
task 3131), were set against the v2.1 soak baseline (§8). v2.2 narrows LOSS,
which can only lower the figures that rest on it (§8). The post-fix
measurement must therefore do one of two things. It can render its rater
prompt from the v2.1 brief text (`git show
c7b682bcb9:fused-memory/calibration/write_triage_rater_brief_v2.md`) and
score with `--rubric v2.1`. Or it can wait for Leo to re-confirm the bounds
against a v2.2 re-rate of the soak's LOSS-consuming pairs. It must not
silently pick up v2.2 from the file. This is a choice for Leo; nothing here
makes it.

**After the embedder cutover (6807)** re-run the whole measurement: the band
edges and every cosine-derived threshold move with the embedder.
The labelled pair set for it is `write_triage_eval_set_v2.md`.

## 8. Soak baseline

The 2026-10-19 soak's 553 adjudicated attaches and the R8 extension, rated
under rubric v2.1. The report is `write_triage_v2_soak_baseline.json`
(rendered as `write_triage_v2_soak_baseline.md`), the scorer's output over
`write_triage_v2_soak_{items.jsonl,verdicts.jsonl,strata.json}`. The sample,
the rating volume and the derivation check are in
`write_triage_v2_soak_derivation.json`. Every figure below is quoted from one
of those two JSON files, or computed from figures in them, unless it is marked
otherwise. Rates and kappas are quoted to four places as the JSON gives them.

Every floor binary meets the kappa floor, and no v1 row that failed the
derivation check fixes an answer (both below), so this baseline meets §4's
conditions for scoring.

These figures are rated under brief v2.1 (commit `c7b682bcb9`), not the v2.2
the brief file now holds. M1 hidden, M1 as amended, M2 restated, and the
hidden_detail and duplicate_clutter harm all rest partly on LOSS. v2.2
narrowed LOSS to needed detail (`write_triage_eval_set_v2.md` §6), so these
figures do not compare with a v2.2-rated measurement until the soak's
LOSS-consuming pairs are re-rated under v2.2.

### Provenance

- **The run** (not in the JSON): a hand-carry by an operator session on
  2026-10-23, per Leo's ruling 254=A on esc-7174-4. It reused the round-0
  ratings that earlier implementer iterations of task 7174 had produced,
  completed two partial shards, and re-rated nothing. Its scratch directory,
  gitignored, is `data/write-triage-v21-rebase-2026-10-21/` (named in the
  step-17 commit's message).
- **Inputs**, read-only: the soak bundle's
  `data/write-triage-soak-2026-10-19/correctness/` (gitignored, main checkout)
  `adjudications.jsonl`, `attaches_soak.json`, `parents.json` (parent text as
  read from Qdrant on 2026-10-19), `hoist.json` and
  `parent_edited_after_attach.json`; the write journal
  (`data/reconciliation/write_journal.db`); and `~/.mem0/history.db` for
  attach-time text. Built by `fused-memory/scripts/build_write_triage_v2_soak_baseline.py`
  with seed 7174, the `shards` default (not in the JSON). The report's
  bootstrap seed, 7157, is the scorer's own `::BOOTSTRAP_SEED`.
- **The R8 extension** (`sample`): restated 40 adjudicated + 60 drawn = 100
  of 125; amended 40 + 60 = 100 of 1,224; contested 473 + 0, the census.
  Unrecoverable: 0 in every stratum (`unrecoverable` is empty), and 0 of the
  20 attach-time pairs (`attach_time_unrecoverable` is empty). The report
  holds 673 items: amended 100, contested 473, restated 100, stored 0.
- **Rating volume** (`rating_volume`, distinct pairs per role): current basis
  primary 673, second 673, third 41; attach time primary 20, second 20. Every
  rater in `shards` is Opus (`opus-primary`, `opus-second`, `opus-third-00`
  to `-02`), so every item is double-rated by Opus on every basis it has. The
  current-basis pairs sit in 17 primary and 17 second shards of 39 or 40, the
  attach-time pairs in one primary and one second shard of 20.
- **Re-asks and third rounds**: `reasked` is 0. Three third rounds, each its
  own rater: `third_00` 41 pairs, `third_01` 2, `third_02` 2. The scorer
  accepts the corpus, so no decisive tie is left. Ties left on binaries the
  derivation does not need, counted with `write_triage_reader_rubric.py::resolve_judgements`
  over the committed verdicts (not in the JSON): home 29, scope 25,
  action_changing 22, loss 2 on the current basis, none at attach time.
- **Every rater answer is committed** in `write_triage_v2_soak_verdicts.jsonl`,
  beside the derived rows of the held v1 rows. A current-basis rater answer on
  a binary a held row fixes carries `non_consumed`. The corpus therefore
  re-scores from git alone, and
  `fused-memory/tests/test_score_write_triage_reader.py::TestTheCommittedSoakBaseline`
  pins that re-scoring it reproduces the report.

### Rater agreement

Cohen's kappa, primary against second, over consumed answers (`kappa`). The
R5 floor is kappa >= 0.7 on at least 60 consumed pairs, for home, conflict and
loss.

| Binary | n consumed | Kappa | Floor met |
|---|---|---|---|
| home | 223 | 0.9462 | yes |
| conflict | 423 | 0.8267 | yes |
| loss | 81 | 0.9212 | yes |
| scope | 286 | 0.844 | yes (report-only) |
| action_changing | 286 | 0.7744 | yes (report-only) |
| digest | 10 | none | no (report-only) |

The floor holds for home, conflict and loss, and each has at least 60
consumed pairs without over-sampling. SCOPE, which the floor does not cover,
is at 0.844 on 286 pairs. DIGEST is computed from the loss spans. All 10
consumed pairs carry one and the same label, NO from both raters (counted from
the committed verdicts, not in the JSON), so chance agreement is 1 and kappa is
undefined: `fused-memory/scripts/write_triage_stats.py::cohen_kappa` returns
None.

### Derivation check

Each v1 row compared with the primary and second raters' current-basis
answers on its binary (`rows`). A candidate row is held, and fixes its
binary, at >= 20 compared answers and >= 0.90 agreement (`check`). Compared
counts answers, two raters to an item. Agreement is agree / compared, rounded
to three places.

| v1 row | Binary | Items | Agree / compared | Agreement | Status |
|---|---|---|---|---|---|
| related_unrelated_home_no | home | 176 | 352 / 352 | 1.000 | held |
| corrects_action_changing_conflict_yes | conflict | 230 | 424 / 458 | 0.926 | held |
| same_subsumed_unrelated_conflict_no | conflict | 20 | 36 / 40 | 0.900 | held |
| same_subsumed_home_yes | home | 18 | 32 / 36 | 0.889 | not held |
| no_new_detail_loss_no | loss | 15 | 25 / 28 | 0.893 | not held |
| corrects_action_changing_scope | scope | 230 | 353 / 424 | 0.833 | not held |
| corrects_noise_conflict_no | conflict | 0 | 0 / 0 | — | not held |
| extends_home_yes | home | 114 | 98 / 228 | 0.430 | retired (R7) |
| corrects_not_action_changing_conflict_no | conflict | 14 | 4 / 28 | 0.143 | retired (R7) |

A row not held fixes nothing, and the raters answer its binary. The two rows
R7 retired are measured and never applied; under v2.1 the raters still
disagree with them as they did under v2.0. same_subsumed_unrelated_conflict_no
holds at exactly the 0.90 bar.

### Headline figures

Sample rate with its Wilson 95% interval, then population rate with its
stratified bootstrap 95% interval.

| Figure | Sample | Population | Reference |
|---|---|---|---|
| M1 hidden in band | 34 / 91 = 0.3736 [0.2812, 0.4762] | 0.3744 [0.2766, 0.4752] | gate 0.15 |
| M1 as amended | 35 / 91 = 0.3846 [0.2913, 0.4873] | 0.3833 [0.2857, 0.4835] | report-only |
| M2 restated correct | 23 / 99 = 0.2323 [0.1601, 0.3246] | 0.2323 [0.1515, 0.3163] | |
| M2 amended correct | 31 / 100 = 0.31 [0.2278, 0.4063] | 0.31 [0.22, 0.4] | |
| M2 contested correct | 322 / 472 = 0.6822 [0.6389, 0.7226] | 0.6822 [0.6385, 0.7246] | |
| M2 stored correct | n 0 | — | |
| M3 hidden corrections | 39 / 371 = 0.1051 [0.0779, 0.1405] | 0.379 [0.2686, 0.4673] | bound unset |
| M4 wrong home | 233 / 671 = 0.3472 [0.3122, 0.384] | 0.5416 [0.4761, 0.6043] | 0.20 |
| M4 direct | 127 / 503 = 0.2525 [0.2165, 0.2922] | 0.4658 [0.3882, 0.5418] | 0.20 |
| M4 via sibling | 104 / 164 = 0.6341 [0.5581, 0.704] | 0.7749 [0.6694, 0.8603] | 0.20 |
| M4 route unknown | 2 / 4 = 0.5 [0.15, 0.85] | 0.5294 [0.0, 1.0] | 0.20 |
| M4 same category | 156 / 445 = 0.3506 [0.3077, 0.396] | 0.5465 [0.4656, 0.6209] | 0.20 |
| M4 cross category | 77 / 226 = 0.3407 [0.282, 0.4047] | 0.5314 [0.4125, 0.6395] | 0.20 |
| M5 conflict-flag precision | 332 / 472 = 0.7034 [0.6607, 0.7428] | 0.7034 [0.661, 0.7442] | |
| M6 drift | 11 of 20 parent-edited children drift | — | report-only |
| Harm per write, sample | triage 0.9553, status quo 0.8525, delta +0.1028 [−0.0268, 0.2377] | | provisional weights |
| Harm per write, population | triage 0.9511, status quo 0.4662, delta +0.4849 [0.396, 0.5742] | | provisional weights |

- **M1**: one band item is UNCLEAR, and 4 items of unknown band are outside
  M1 (`band_unknown`). The band holds 91 restated writes and 1 contested one
  (counted from `items.jsonl`, not in the JSON).
- **M2** per-project counts are in the JSON's `by_project`.
- **M3 by ending**: restated 25, amended 14, contested 332, stored_unlinked 0,
  stored_linked 0. The 39 hits are the restated and amended endings. The
  sample rate is dominated by the contested census, 332 of 371. The
  population rate is dominated by the 14 amended conflicts: each amended item
  stands for 12.24 amended writes (1,224 / 100), each restated item for 1.25
  (125 / 100).
- **M4** counts every decided attach, contested ones included, and leaves 2
  UNCLEAR out. Its population rate sits above the sample rate because the
  amended stratum is weighted 12.24 per item.
- **M5 by SCOPE**: central 158, secondary 140, unknown 34. **By
  action_changing**: yes 273, no 32, unknown 27. Each unknown holds the 10
  contested pairs that `unreached_follow_ups` lists for scope and
  action_changing, where held corrects_action_changing_conflict_yes fixed
  CONFLICT YES over the raters' NO, so no rater was asked the follow-up.
  The rest of each unknown is ties: scope 24, action_changing 17 (counted
  from the committed verdicts, not in the JSON).
- **M6**: 20 children had their parent edited after attach, 20 were rated at
  attach time, 0 are undecided there, and 11 drift (`drifted_entry_ids`).
- **Harm** is under `PROVISIONAL_HARM_WEIGHTS`. Triage's cells: wrong_home
  233, false_conflict 140, hidden_conflict 25, digest_conflict 14,
  hidden_detail 13, attached_supersession 12. The status quo's: orphaned_conflict
  371, duplicate_clutter 31. Triage costs more per write than the status quo
  in both regimes. The sample interval spans 0 and the population interval
  does not.

### The counterfactual: F3′ alone

M1 as amended reads every band attach as if it were amended. It is 0.3846 on
the sample and 0.3833 on the population, and both intervals lie wholly above
0.15. F3′ (task 7155), though, sends a band write to a judge that may amend it
or contest it, and a contested write is visible to M1. The counterfactual is
therefore the case in which the judge contests none of them, and what F3′ can
reach depends on how many band CONFLICT writes it contests. The split of the
35 hits, counted from the committed corpus with the scorer's own derivation
(not in the JSON), gives the bound. 24 hits rest on CONFLICT, which an
amendment still hides in an untagged digest. One of them,
`998b9d5d-a30a-4832-99de-6978736dfcf8`, is already contested and counts as
hidden only inside the counterfactual. The other 11 rest on HOME with LOSS,
and in all 11 DIGEST is NO: the needed detail lies past the first 240
characters, so amending them hides it as well. With only those 11 hidden, M1
would be 11 / 91 = 0.1209. The gate allows at most 13 hidden writes of 91:
13 / 91 = 0.1429 passes and 14 / 91 = 0.1538 fails (computed from the JSON's
n of 91 and the 0.15 gate). On this evidence F3′ alone clears the M1 gate only
if its judge contests at least 22 of the 24 band CONFLICT writes. A digest or
render change for the detail hits alone cannot clear it: with every CONFLICT
write still hidden, M1 would be 24 / 91 = 0.2637. These are sample counts; the
population rate is not split here.

### Caveats

- The soak holds no stored items, so the stored cells, M2 stored and M3's
  stored endings are unmeasured (n 0).
- The status quo's earlier records are only each write's parent (every item's
  `earlier_ids` is empty, in `items.jsonl`). Its orphaned_conflict and
  duplicate_clutter counts are a lower bound on its harm, and the harm delta an
  upper bound on triage's excess.
- The strata weights (`strata`): restated 100 sampled of 125, amended 100 of
  1,224, contested the census of 473. Population rates of any figure that
  counts amended writes (M3, M4, harm) rest on 100 amended items standing for
  1,224.
- Two items are UNCLEAR and left out of every rate (`unclear_entry_ids`):
  `756c582f-ba18-4d51-88e3-5615cfa6c8cf` (contested) and
  `adf43f4e-0405-4c5c-aec4-199fcd617d2e` (restated). On each, held
  same_subsumed_unrelated_conflict_no fixed CONFLICT NO where both raters said
  YES, so no rater reached LOSS (`unreached_follow_ups`). Leo's ruling of
  2026-10-23, recorded on esc-7174-4 (not in the JSON), resolves such a
  follow-up UNCLEAR (§4). A rater row whose own HOME is YES and CONFLICT NO
  must give LOSS, or the rubric refuses it
  (`write_triage_reader_rubric.py::DECISIVE_FOLLOW_UPS`).
- One binary in the corpus was agreed by the primary and second raters and
  then overturned. A third rater answers the whole item blind, and every vote
  counts toward the majority. On `85c91566-7979-42c3-adbd-b37243d06530`
  (restated, in the M1 band) the primary and second agreed HOME YES and split
  on LOSS, primary NO and second YES. All three third rounds answered HOME NO,
  so none of them gave LOSS, and HOME resolved NO 3 to 2. The item is
  wrong_home and not M1-hidden. Had HOME resolved YES and LOSS YES, M1 would
  be 35 / 91. Leo ruled on 2026-10-23 to keep the all-votes majority. The
  votes are counted from the committed verdicts; the ruling is not in the JSON.

### v2.0 history

Task 7157 first scored this soak under rubric v2.0 (553 items, seed 7157).
Its headline figures, quoted from git and not from the current JSON: M1 0.46;
M2 restated 0.33, amended 0.63, contested 0.67; M3 0.04 sample and 0.36
population; M4 0.23; M5 0.69; harm delta −0.07 sample and +0.24 population.
They are superseded because that baseline was scored through a kappa floor
miss, with no binary at the floor on the corpus basis, and through two failed
v1 derivation rows, EXTENDS → HOME YES and CORRECTS not action-changing →
CONFLICT NO, which R7 then retired. git holds the full v2.0 section:
`git show 076c1ad8f6:fused-memory/calibration/write_triage_reader_metrics.md` (task 7157).
