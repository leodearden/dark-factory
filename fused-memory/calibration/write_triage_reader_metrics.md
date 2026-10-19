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
item weighted N_h / n_h, and a stratified bootstrap 95% interval. An item whose
deciding binary resolved UNCLEAR is left out of n and counted as `unclear`.

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
- About 10 blind anchors per shard detect drift between batches.

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

To be filled when the soak's 553 adjudicated items are rated under v2 (task
7157 step 16).
