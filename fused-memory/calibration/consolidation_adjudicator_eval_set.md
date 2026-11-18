# Consolidation adjudicator evaluation set

The labelled set that `plans/consolidation-adjudicator-prd.md` (K5, E7) scores the consolidation
adjudicator against. The build scripts, logs and raw responses named below are provenance; they
live in the untracked evidence directory `plans/detector-options-10244-2026-11-17/adjudicator-eval/`
of the main checkout and are not needed to use the set. The test split must never be shown to an
adjudicator prompt outside the service's own calibration run.

`consolidation_adjudicator_eval_set.jsonl` holds 183 cases, one JSON object per line: 63 in the dev split and 120 in the test split.

| split | fold | contaminated | must_not_fold | real_proposal | total |
|---|---|---|---|---|---|
| dev | 29 | 14 | 13 | 7 | 63 |
| test | 45 | 45 | 30 | 0 | 120 |

How the file was built:

- `build_eval.py` built the 153 dev, fold and contaminated cases.
- `extend_eval.py` then added the 30 test must_not_fold cases (T-M01..T-M30). It also added `n_members` and `mixed_category` to every case.
- Apart from those two new fields, every earlier case is unchanged from the first build (`eval_set.v1.jsonl`, sha256 `8d2f8427…`); `verify_eval.py` checks this.
- Everything is checked by `verify_eval.py`, which reports 0 failures in `verify_log.json`.
- Build figures are in `build_log.json` and `extend_log.json`.

Fields and spend:

- `mixed_category` is true for exactly 7 cases, all in the dev split: A03, A05, B03, B06, B12, B14 and E6.
- sha256 `df2723389d1ee696b3b5bf3e7efa4c1e558263ca08669c7e58fd8c5befc135af`
- API spend: **$0.029756** in total, from 120 gpt-6-luna calls (low effort), all for test-split claims:
  - $0.022884 for fold and contaminated claims, against a $0.15 cap.
  - $0.006872 for must_not_fold claims, against an extra $0.05 cap.

  The ledger is `spend_ledger.jsonl` and the raw responses are in `responses_claims.jsonl`. No adjudicator was called, and no other model saw the test cases.

## Counts

### Test split, fold and contaminated (90 cases)

| kind | project | category | cases |
|---|---|---|---|
| fold | dark_factory | procedural_knowledge | 15 |
| fold | dark_factory | preferences_and_norms | 3 |
| fold | dark_factory | observations_and_summaries | 2 |
| fold | reify | procedural_knowledge | 12 |
| fold | reify | preferences_and_norms | 5 |
| fold | reify | observations_and_summaries | 8 |
| contaminated | dark_factory | procedural_knowledge | 17 |
| contaminated | dark_factory | preferences_and_norms | 4 |
| contaminated | dark_factory | observations_and_summaries | 2 |
| contaminated | reify | procedural_knowledge | 10 |
| contaminated | reify | preferences_and_norms | 5 |
| contaminated | reify | observations_and_summaries | 7 |

The 45 fold and 45 contaminated cases were all built. There is no shortfall.

Member counts:

- **Fold cases:** 8 cases have 3 members, 10 have 4, 3 have 5 and 24 have 6.
- **Contaminated cases** (peers plus the foreign record): 19 cases have 3 members, 15 have 4, 9 have 5 and 2 have 6.

Hazards: 27 `member_carries_correction_banner`. No `mixed_category` hazards occur, by construction.

### Test split, must_not_fold (30 cases, every case has 3 members)

| project | category | cases |
|---|---|---|
| dark_factory | procedural_knowledge | 14 |
| dark_factory | preferences_and_norms | 2 |
| reify | procedural_knowledge | 9 |
| reify | preferences_and_norms | 3 |
| reify | observations_and_summaries | 2 |

All 30 cases were built. There is no shortfall.

The projects are split 16/14, in proportion to the seed population (dark_factory 165 peers, reify 136).

Hazards: 6 `member_carries_correction_banner`.

Overlap:

- The 90 member slots hold 68 distinct ids.
- **62 of those ids (80 of the 90 slots) also appear in test fold or contaminated cases.**
- The 30 seeds are all distinct.

### Dev split (63 cases, pilot cases.jsonl unchanged)

| kind (pilot kind) | dark_factory | reify |
|---|---|---|
| fold (A_curator_folded 17 + D_live_good_fold 12) | 7 | 22 |
| contaminated (B_contaminated 14) | 0 | 14 |
| must_not_fold (C_must_not_fold 13) | 10 | 3 |
| real_proposal (E_real_proposal 7) | 7 | 0 |

How the project was assigned to dev cases:

- **A and B cases:** reify. The cases come from the 3130 fixture, whose calibration report (`fused-memory/calibration/write_triage_calibration_report.md`) states `project_id: reify`.
- **C and D cases:** the project in the packet key.
- **E cases:** dark_factory, where the members resolve.

Dev-split categories:

- 4 fold, 4 contaminated and 1 real_proposal case mix categories.
- The rest are mostly procedural_knowledge.

## How expectations are mapped

- **Test fold:** `decision_set ["proceed"]`, `keep_ids` = all members, `foreign_id null`.
- **Test contaminated:** `decision_set ["proceed","drop"]`, `keep_ids` = the group's peers, `foreign_id` = the foreign record. A response counts as correct only if it does not keep the foreign id.
- **Dev expectations** are mapped from the pilot:
  - `proceed` → `["proceed"]`
  - `drop` → `["drop"]`
  - `proceed_without_foreign_or_drop` → `["proceed","drop"]`
  - `drop_or_unresolved` → `["drop","unresolved"]`

  The pilot's `keep_ids` and `foreign_id` carry over. `reason_code` carries over too: E6 keeps `cross_category` with `["drop"]`. The pilot's `min_keep_fraction` (2/3) also carries over on A, D and the E proceed cases.

## Construction rules as executed (test split)

1. **Live snapshot.** Every topic-bearing record in fused_dark_factory and fused_reify was scrolled, with its vector: 3,319 and 2,148 records. The collections were only scrolled and retrieved; nothing was written.
2. **Curation-sitting topics.** A topic counts if it has at least one `canonical: true` record and the topic was classified by step0's rule (`live/step0.py::role`, `curated`, and the `group_kind` order). The order matters: a Stage-1 consolidator canonical wins over a curation-sitting one. This gave 57 dark_factory topics and 68 reify topics.
3. **Peers.** A peer is a record that satisfies `store.is_peer`: it has a topic, is not canonical, has no parent_id, is not superseded and is not a consolidation-kind record. Canonicals are never members.
4. **Group size.** A group needs 3–12 peers in total across all categories. Only the majority category is kept (ties go to the alphabetically first category), and that category must hold at least 3 peers. 21 of the 45 fold groups spanned categories and were trimmed this way.
5. **Exclusions.** Excluded record ids came from three sources:
   - every UUID in the pilot cases' members and audits (210)
   - every memory_id and cluster_id in `write_triage_calibration.jsonl` (104)
   - every UUID anywhere in `live/packet_key.json` (735)

   That is 856 ids after deduplication.

   Excluded topics came from three sources:
   - pilot topics
   - packet topics (`topic` and `candidate_topic`)
   - the live topic of every excluded id that still resolves

   That is 94 topics.

   A group is dropped if its topic is excluded, or if fewer than 3 majority-category peers remain after id exclusion.
6. **Selection.** The 52 eligible groups were sorted by (project, topic) and shuffled with `random.Random(20261117)`. The first 45 became fold cases: up to 6 peers each, sampled at random.
7. **Contaminated cases.** These used the remaining 7 groups first. The population was not large enough for 45 distinct groups, so **38 contaminated cases reuse a fold group**. Each reused case draws a different member sample, and the code checks that the sample differs from the fold case's set. Each case takes 2 to 5 peers at random (never more than the group holds) plus one foreign record.
8. **Foreign record.** The foreign record is the peer, in the same project and category, with the highest cosine to the normalised centroid of the chosen peers. Candidates are peers of any other curation-sitting topic of any size, excluding excluded ids and topics, and they come from that topic's stored vectors.
   - Cosine to the centroid: min 0.457, median 0.663, max 0.913. 6 cases have a cosine of 0.80 or more.
   - 3 foreign records are each used in 2 cases.
9. **Shuffling and hazards.** Members are shuffled. Hazards use `CORRECTION_BANNER_RE`, imported from `fused_memory.reconciliation.consolidation_auto`, run on the full text. Text is cut at 4,000 characters, and a cut is marked with a trailing `" [cut]"`.
10. **Claims.** Claims use the pilot's exact prompt: `gen_claims.INSTRUCTIONS`, `render`, `SCHEMA`, `gpt-6-luna` at `low`, with 4,000 max output tokens. The contaminated cases' foreign members were shown to the claim model. All 90 calls completed. 7 test claims (and 4 dev claims) exceed the prompt's 200-character limit; they were left as generated, which matches the pilot.

## Construction rule for must_not_fold test cases (extend_eval.py)

1. **Population.** These cases use the same 52 post-exclusion eligible groups and the same exclusions as above. Only peers in each group's majority category count.
2. **Usable project-category pairs.** A (project, category) pair is usable only if it holds at least 3 groups. Every pair qualifies except dark_factory observations_and_summaries, which has 2 groups and was skipped.
3. **Seed draw.** Seed candidates are every peer of the usable pairs. Each project's candidates are sorted, then shuffled by a fresh `random.Random(20261117)` stream.
4. **Quotas.** Each project's quota is proportional to its seed population: dark_factory 16, reify 14. Seeds are taken in shuffled order and never reused. A seed is skipped if its triple duplicates an earlier one; no seed was skipped.
5. **Neighbours.** Every candidate peer of the same project and category, but a different topic, is ranked by stored-vector cosine to the seed. The case keeps the nearest peer of the nearest other topic, then the nearest peer of the next topic. The result is three topics.
6. **Expectation.** `decision_set ["drop","unresolved"]`, `keep_ids []`, `foreign_id null`.
7. **Provenance.** Each case records `seed_id`, `seed_topic`, `topics` (all three), `neighbour_ids` and `neighbour_cosines_to_seed`.
8. **Claims.** Each case gets one gpt-6-luna call with the pilot's exact prompt, shown the three members. No claim exceeds 200 characters.

Neighbour cosines to the seed:

| neighbour | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| nearest other topic | 0.503 | 0.581 | 0.640 | 0.685 | 0.790 |
| second other topic | 0.468 | 0.505 | 0.570 | 0.612 | 0.752 |

## Eligible population per project and category

Counts are curation-sitting groups with 3–12 peers and a majority category of at least 3, before and after exclusions.

| project | category | groups before | peers before | dropped (topic excluded) | groups after | peers after |
|---|---|---|---|---|---|---|
| dark_factory | procedural_knowledge | 30 | 203 | 10 | 20 | 138 |
| dark_factory | preferences_and_norms | 5 | 27 | 0 | 5 | 27 |
| dark_factory | observations_and_summaries | 2 | 15 | 0 | 2 | 15 |
| reify | procedural_knowledge | 23 | 143 | 11 | 12 | 73 |
| reify | preferences_and_norms | 5 | 30 | 0 | 5 | 30 |
| reify | observations_and_summaries | 10 | 41 | 2 | 8 | 33 |
| **total** | | **75** | **459** | **23** | **52** | **316** |

No group was dropped by id exclusion alone. Every group that held an excluded id was already dropped through its topic.

## Verification (verify_eval.py, 0 failures)

- **File shape.**
  - The file parses line by line: 153 lines.
  - Field order matches the schema.
  - Case ids are unique.
  - Every case has at least 2 members and a non-empty claim.
- **Test members.**
  - Every test member id resolves live in its case's project.
  - The cut text and category of every test member equal the live store at verify time.
  - Every kept id and every foreign id is a live peer.
  - Kept ids all carry the case's topic.
  - No canonical is a member.
  - Hazards recompute identically.
- **Leakage.**
  - No test id is in the dev split, the pilot, the packet or the 3130 fixture.
  - No test member's topic is a pilot, packet or dev topic.
- **Foreign records.** Every contaminated case's foreign record carries a topic that differs from its group's topic.
- **Dev split.** Members, claim, hazards and keep_ids are byte-equal to the pilot `cases.jsonl`.
- **must_not_fold cases.**
  - Each has 3 members with 3 different live topics, all in one category, and all live peers.
  - Provenance topics and ids match the members.
  - Every member resolves live with matching text.
  - No member id or topic appears in dev, the pilot, the packet or the 3130 fixture.
- **New fields.**
  - `n_members` and `mixed_category` are consistent with the members.
  - The `mixed_category` set equals {A03, A05, B03, B06, B12, B14, E6}.
  - Every first-build case is unchanged apart from the two new fields.
- **Dev members that no longer resolve.** Dev members were not required to resolve, and they were not edited:
  - None of the members of the 31 A and B dev cases (from the 3130 fixture) resolve live any more: 144 member slots, 79 distinct ids. Only 16 of the fixture's 104 ids are still live.
  - Every C, D and E dev member resolves.
  - No dev member's text has drifted.

## Label caveats

- **Fold labels are a noisy positive.** A group sharing a curation-sitting topic is not proof that it should be folded whole. In this session's blind rating, raters called only 10 of 16 existing topic groups foldable as a whole, and those 16 mixed sitting-curated and agent-tagged groups. **0 of the 45 test fold cases** come from those 16 topics (the packet's positive controls), because the exclusions remove them. Expect some test fold groups to be honestly splittable; an adjudicator that shrinks a fold case is not necessarily wrong. The pilot's `min_keep_fraction` was not added to test cases, because the brief sets keep_ids = all members.
- **Contaminated labels rest on the topic boundary.** The foreign record belongs to a different curated topic, which makes it foreign by topic stamp, not by content judgement. The 6 cases whose foreign record has a cosine to the centroid of 0.80 or more are the most likely to hold a foreign record that genuinely teaches the same lesson: TC cases with `provenance.foreign_cosine_to_centroid` ≥ 0.80.
- **must_not_fold labels rest on curator topic boundaries.** A curator may have kept two topics separate that are still one lesson, so a small share of these labels may be wrong. The cases most at risk are those whose nearest neighbour has a cosine to the seed of 0.72 or more: 4 cases, with nearest cosines 0.7245, 0.7279, 0.7804 and 0.7902 (see `provenance.neighbour_cosines_to_seed`). These cases also share 62 member ids with the fold and contaminated cases. A model that memorises members across cases is not a concern for a one-shot adjudicator, but per-kind results are not independent samples.
- **Reuse.** 38 of 45 contaminated cases reuse a fold group's topic with a different member sample, so the fold and contaminated results are not independent across those topics.
- **Sitting labels come from a heuristic.** "Curation sitting" means step0's agent_id and provenance-key rule, not a sitting log.
- **Small categories.** dark_factory observations_and_summaries has 2 fold cases, so per-category rates on the test split will be thin outside procedural_knowledge.

## Files

- `consolidation_adjudicator_eval_set.jsonl`: the set.
- `build_eval.py`: the build.
- `extend_eval.py`: the must_not_fold extension and the two new fields.
- `extend_log.json`: extension figures.
- `eval_set.v1.jsonl`: the first build, kept as the baseline for the unchanged-case check.
- `verify_eval.py`: the checks.
- `build_log.json`, `verify_log.json`: build figures and check results.
- `responses_claims.jsonl`, `spend_ledger.jsonl`: raw claim responses and the spend ledger.
