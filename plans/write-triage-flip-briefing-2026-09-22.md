# Briefing: esc-3169-1 — the write_triage_enabled flip gate

Investigated 2026-09-22 against main `0583aa060f`. Nothing was flipped, resolved, stamped or edited. The reify fixture (104 records, 20 clusters) is the population for every accuracy or recall figure unless stated; every derived figure carries its rule.

## 0. The short version

Write triage is fully built and wired; the flip predicate passes; all three dependencies (3128, 4762, 4811) are done. What holds it is Leo's 2026-08-27 HOLD, reaffirmed 09-21, and the escalation-watcher's 09-18 addendum concluding "retrieval is the binding constraint" with `judge_candidate_count` 5→10 as the one knob with headroom.

This investigation found the composite (~0.28) measures the wrong thing. The judge's verdict is filed against the band's rank-1 record unconditionally, so a correct attach is bounded by recall@1 (0.27), not recall@5 (0.51). Widening the slate, the addendum's recommendation, buys nothing until the verdict names its candidate, and that fix, "option (a)", currently has no owner. Separately, the judge sees at most 1200 characters of each candidate while every canonical in the fixture is longer (median 5617), so both the retrieval loss and the judge loss trace to long canonicals. The cheapest untried experiments cost cents. The status quo the hold protects catches essentially nothing (the e2 replay's best score was 0.899 against a 0.92 threshold). And leaf ζ, the standing objection, has been dispatchable since 08-20.

Jev bears on one sub-problem (the judge's output contract and latency), not on retrieval; a local cross-encoder reranker gives the same binding without a vendor and is the comparison Jev has to beat.

## 1. The system

**Chokepoint.** `add_memory` in `fused-memory/src/fused_memory/server/tools.py`. It derives `allow_near_duplicate`, then `triage_enabled = category ∈ {procedural_knowledge, preferences_and_norms, observations_and_summaries} and write_triage.enabled`. Graphiti-primary categories and `category=None` writes are in neither world: never guarded, never triaged. The flag is global to the fused-memory server: it applies to every hosted project, while every number below is from reify's corpus; dark_factory's 49,628 records have never been measured against these thresholds.

**Flag off (today).** Two reject guards, both skipped by `allow_near_duplicate: True`, no agent-class exemption (the recon-stage exemption was retired by task 3134):
- Topic-cluster guard: phrase match against five config-seeded dark-factory topics; procedural + preferences; error `ProceduralKnowledgeKnownTopicClusterWriteRejected`.
- Cosine guard: procedural_knowledge only; one same-category search, limit 5; reject at store_score ≥ 0.92 (schema default, no config override); error `ProceduralKnowledgeNearDuplicateWriteRejected`.
- observations_and_summaries is guarded by nothing.

**Flag on.** `dup_guard_base_exempt = not triage_enabled and not allow_near_duplicate` switches both reject guards off for every triaged category, and `triage_write` runs:
1. Force-store before retrieval: `allow_near_duplicate: True` (its meaning changes from "don't reject me" to "don't reroute me"), or caller metadata containing any of `parent_id`, `kind`, `x_contested` by presence (~5% of records carry a `kind`).
2. Retrieval: one Mem0 search across the three Mem0 categories, `candidate_k = 20`, `anchor_topics=False`.
3. Band on the max cosine over the 20, pooled thresholds only: `≥ t_high 0.8868` → `restated` against the argmax (a child winner is hoisted to its parent), no LLM; `t_low 0.5230 ≤ s < t_high` → judge; `< t_low` → `stored`. `t_high_by_category` has no reader on the write path. Note t_high is literally an observations pair's score, and 18% of unrelated pairs sit above t_low, so with a max over 20 neighbours from a 49k corpus the middle band is nearly every write: a synchronous LLM call (10s timeout) on almost every triaged write.
4. Judge: gpt-4o-mini via openai (inherited from `llm.*`), 64 max tokens. Slate = top 5 by cosine, band winner always included. Closed answer `distinct | restates | amends | contests`. Any failure → fail-open `stored`, counted; 10 in an hour files a blocking **L1** escalation. Fail-open is indistinguishable from "novel" in the ack stream by design; the L1 queue is the only signal.
5. Attach: `restated` → child `kind=sighting`, only counted in grouped reads; `amended` → child `kind=amendment`, body digested (240 chars) into the canonical's grouped read; `contested` → amendment plus `x_contested: True`, which nothing consumes (no gate, no escalation). The canonical is never edited. The child keeps its own category, so grouped documents can mix categories.
6. Ack gains `routed` and `canonical_id`.

**What the flip does not deliver.** The PRD band rule "s < T_high with a topic-cluster hit still goes to the judge" is unimplemented. The DEFERRED comment in tools.py gives two premises: the judge is a stub (false since 08-26, the real judge is wired at `judge=judge_write`) and ζ lands before 3169 (still open; the comment anticipates "or still named here"). There is no runtime topic-cluster store; only the five config seeds.

**Hot reload.** `write_triage.enabled` and all judge leaves are green-tier and read live per write. Reload, not restart.

**Text an operator would read that has gone stale.** config.yaml's "until then the middle band is answered by a deliberate stub" is a satisfied conditional; the `judge_enabled` comment below it is current. The escalation's own detail still says "no before_done" and "restart"; the task description was corrected in place, the escalation detail is write-once. The `candidate_k` comment's recall series (26.1%@5 … 88.5%@50, "the live corpus") is one cluster, n≈15, dark_factory, 2026-08-02, repeated in schema.py, write_triage.py and a test.

## 2. The problem(s)

**Built to solve** (PRD `memory-write-path-convergence.md`, from the 2026-07-27 reify curator RCA): the corpus accretes instead of converging. 89 defective entries from ~40 one-shot agents; the reject guards could not have caught them (genuine rediscoveries at cosine ~0.82 vs 0.92; five hand-fed topics). Fix: cosine as candidate generator, judge for the middle band, non-destructive attach, grouped reads, then invert the writer instruction (ε, 3131) to "write freely; the server deduplicates".

**What blocks it on the record.** The HOLD's surviving grounds: composite ~0.27; ζ unlanded so the topic guard goes off with no replacement; contested unmeasurable. The 09-18 addendum adds: retrieval binding, widen the slate.

**What this investigation adds.**
- **P1 Attach-target binding.** The system prompt (written by 4811 on 09-17) says: decide "restates"/"amends" if ANY candidate makes the same claim. The user prompt (4762's marker, landed 09-16) marks one candidate `attach_target` and says the verdict is filed against it. `triage_write` line 898 then files it against `decision.canonical_id`, the rank-1 record, unconditionally. Two tasks wrote the two instructions eight days apart; neither reconciled them. A canonical at rank 2–5 yields either a miss or an attach to the wrong record; the "ANY" rule makes wrong-record the likelier reading, and it is unmeasured because the eval seeds the canonical at slot 0 and marks it target. The predicate passes on option (b) and prints its own warning that it "does not assert that the bound candidate is CONSUMED". Option (a), a verdict that names its candidate, was "4798 item 7"; 4798 was coalesced into 5277 on 09-09 and neither carries it. **Option (a) has no owner.**
- **P2 Elision.** `_FIELD_CHARS = 1_200` cuts every candidate and the new entry (`f2764d7e4f`, 08-26, so both measurements were equally elided). 20/20 canonicals exceed it; all 75 duplicate cases' canonicals exceed it (median 5617, 73/75 > 2400); 15/75 duplicate entries exceed it too. Token-set overlap (`[a-z0-9_]+`, len>2, |dup∩canon|/|dup|): median 0.85 against the full canonical vs 0.37 against its first 1200 chars; 44/75 cases lose more than half. CAVEATS never mentions it. A plausible mechanism for the 27 `stored` misses; untested.
- **P3 Recall is overstated.** 0.508@5 excludes 21 of 84 records (three clusters whose canonicals rotated after the session; one is the largest and longest cluster, the other two mid-pack). Counting them as misses: 0.20@1, 0.38@5, 0.54@10. Recall@20, the production width, was never measured.
- **P4 Soft controls.** Distractors are a deterministic rotation of cross-cluster records, not nearest neighbours (hard-negative median cosine 0.785 vs unrelated 0.425). 18/18 says nothing about false attaches on production slates. Distinct n=3, pseudo n=6.
- **P5 Regime change for two categories.** observations goes from unguarded to fully triaged on a pooled t_high its own pairs could not calibrate (hardest measured negative 0.0025 below t_high, n=6). preferences has zero same-category negatives measured.
- **P6 Root cause shared.** The retrieval loss (long canonical = least retrievable member, the PRD's own pathology) and the judge loss (long canonical = truncated) have the same cause. Multiplying them as independent overstates independence and understates what one storage-shape or prompt change would move.

## 3. What we tried, and how each turned out

| Landed | What | Outcome |
|---|---|---|
| 2026-07-29 | α 3130 fixture + calibrator (`b5d9811d64`) | t_high 0.8868, t_low 0.5230, 0 deterministic FPs; 7/301 dup pairs above t_high; recall 17/25/32/45 of 63 at k=1/3/5/10 |
| 2026-08-02 | Retrieval-width study, esc-3181 cluster, dark_factory | same-category recall 26.1@5 … 88.5@50, one cluster; `candidate_k=20` chosen from it |
| 2026-08-02/03 | 3357 per-category cutoffs | procedural 0.839; obs refused (4/51 pairs); pref refused (28/0) |
| 2026-08-18/19 | δ 3129 child records + grouped reads; θ 3133 `consolidate_memories` | landed |
| 2026-08-23 | 3111 topic-anchored pin at the search seam | landed; write_triage opts out (`anchor_topics=False`) |
| 2026-08-25 | β 3127 triage skeleton (`3b3e6a9b8d`) | fail-open, bands, storm counter |
| 2026-08-26 | γ 3128 judge, 5 review cycles (`3cd22a8bd6`) | duplicate 40/75 = 0.533 (31 stored), distinct 2/3, pseudo 5/6, distractor 18/18, contested FP 5 |
| 2026-08-27 | Leo HOLD; 4810/4811 filed | five grounds on 3169 metadata |
| 2026-08-28 | 4822 attach-target contradiction plan + slate rescue fix (`7b0de723c0`) | landed |
| 2026-08-30 | κ 3136 scheduled dedup report ruled FAIL (esc-3136-3) | lexical recall 0/301; ANN 25/301 with inverted AUC 0.25; the 0.92 guard catches 1/301; successor 4916 |
| 2026-09-05 | 4810 behavioural predicate (`19811b1acd`) | predicate passable |
| 2026-09-07 | ι 3134 Stage-1 rewire, recon exemption retired | landed (task's `done_provenance.commit` hash is wrong; real merge `bb633fe0f9`) |
| 2026-09-16 | 4762 option (b) prompt marker + eval determinism fixes, via carrier train `3694de200c` | landed; verdict still carries no candidate id |
| 2026-09-17 | 4811 exemplars + rule inversion, corpus-wide re-measure (`f16fb39bb5`, merged `89e37fd6fb` 09-22) | duplicate 41/75 = 0.547 (+1), distinct 1/3, pseudo 5/6, distractor 18/18, contested FP 8, split {10,30}→{1,40}; fixture expansion ratified undeliverable (esc-4811-3) |
| 2026-09-18 | L2 watcher addendum (auto-memory, `project_task_4762_landed_via_carrier_2026_09_16.md`) | "retrieval is the binding constraint"; recommends `judge_candidate_count` 5→10 (~0.28→~0.39) |
| 2026-09-21 | Leo re-scopes rationale #4; 5547 filed, by ruling not a 3169 dep | HOLD unchanged |

Proposed and never tried: a judge model above gpt-4o-mini and a wider slate (esc-3128-1, 08-26); a cross-encoder or pairwise adjudicator (3136, 08-11). Never raised: the elision; recall@20 on the fixture; a per-case verdict dump; hard distractors.

## 4. What is still planned, and what it buys

| Item | Status | What it buys |
|---|---|---|
| ζ 3135 (+4493) topic-cluster auto-seed / runtime store | **pending and dispatchable since 08-20** (its only dep 3133 is done); the "blocked on 3133" note on 3169 is stale | The replacement for the guard the flip switches off; the substrate for the unimplemented topic-hit→judge rule |
| 5547 dark_factory fixture + calibration report | pending, not a 3169 dep | Per-category cutoffs for the audit script. For the flip: confidence only, unless the judge eval is also run on it, in which case short retained peers could change the decision |
| Option (a), verdict names its candidate | **no owner** (4798 → 5277, neither carries it) | Lifts the strict ceiling from recall@1 to recall@5 and removes the wrong-record class; the precondition for any slate-widening to pay |
| 5277 A4 eval checkpointing / per-case dump | pending | Band-restricted split and elision test fall out of one cheap run |
| ε 3131 writer inversion | pending, dep-gated on 3169 | Correct only once triage is live |
| Reject-guard family 3430 (done) / 4738 (done) / 4729, 4773, 4179, 4255, 4739 (deferred) | fallback while the hold stands; do not cancel | 4179 carries a self-cancel-if-flipped instruction, not yet active |

## 5. Options from here

Rule for estimates: strict correct-attach = d + (r1 − d)·j; any-attach = d + (r5 − d)·j; d = deterministic share (7/301 pairs, a pair rate; per-record anywhere in [0, 7/75]), r1 = 17/63 = 0.27 (0.20 if absent counted), r5 = 32/63 = 0.51, j = 41/75 = 0.547. Strict 0.15–0.19 (0.12 at r1 = 0.20); any-attach ~0.29; the difference ~0.13 is an upper bound on wrong-record attaches, unmeasured. Flag-off cosine guard: e2 replay 0/12 with max observed 0.899 vs 0.92; ≤7/75 records by the fixture. Topic guard: unmeasured, small.

**A. Hold as-is.** Nothing changes; corpus keeps accreting; ε stays blocked. The hold compares ~0.16 strict against a status quo near zero, and every day of hold is a day of writes that will need consolidation later.

**B. Flip now.** Strict ~0.16; up to ~0.13 of duplicate writes attached to the wrong canonical with `amended` bodies digested into the wrong grouped read; obs triaged on an uncalibrated pooled threshold; topic guard off with no replacement; contested markers nobody reads; an LLM call on nearly every triaged write; `allow_near_duplicate` and the reject error change meaning for any caller expecting them. Reversible in one reload; wrong attaches are re-parentable by hand via `update_memory metadata_patch`, nothing detects them; writes made while on are not undone by a reload. Not recommended on this evidence, though the gap to the status quo is smaller than the record implies.

**C. Fix the instrument, then decide. Cents, about a day.** (i) Add a per-case verdict dump to the eval. (ii) Patch the calibrator's `search_fn` to pass `categories` and `anchor_topics=False` like production (3111's pin landed after the 08-03 run and would otherwise inflate a re-run), then measure recall at k=20 with the 21 rotated canonicals resolved; embeddings only. (iii) Re-run the judge with `_FIELD_CHARS` raised to cover whole canonicals, ~102 gpt-4o-mini calls. (iv) Run it with nearest-neighbour distractors and without seeding the target at slot 0, which produces the first honest wrong-record and false-attach numbers. Reversible; this is the option the evidence most supports, and it is the prerequisite for measuring anything else, Jev included.

**D. Own option (a), then C.** File it (5277 or new): judge returns `{verdict, candidate_id}` validated against the slate, threaded through `BandDecision`; reconcile the two contradicting prompt instructions in the same change. With gpt-4o-mini this is a structured-output change, about a day. Then the addendum's slate widening becomes real: ceiling 0.51 at 5, 0.71 at 10.

**E. Dispatch ζ (3135) now.** It is the ruling's standing objection and has been dispatchable for 33 days. Independent of everything above.

**F. Widen the slate alone.** The addendum's recommendation. Without D it raises any-attach but not strict correct-attach, and adds hard neighbours to the slate. Do after D, with recall@20 measured first.

**G. Change the judge model alone.** Measured under the current harness it inherits the elision and slot-0 seeding; do after C.

**H. Cross-encoder / pairwise reranker over the 20** (proposed 08-11, never tried). Scores each (entry, candidate) pair independently: candidate-bound by construction, local, no vendor, no option-order effects. Gives P1's fix and a wide slate at once; new model dependency on the synchronous write path, latency per write. Worth a fixture measurement after C.

**A staged flip with the judge off** was considered and dropped: detection would be the deterministic band alone (7/301 pairs), the same ~2% as the guard it replaces, and a procedural-only flip needs code (`_TRIAGED_CATEGORIES` has no knob) and still uses the pooled cross-category t_high.

## 6. Jev (typesafe.ai)

**What it is** (vendor and third-party claims, launched 2026-09-15, early access, $40M seed, no technical report or weights published): a non-autoregressive "System One" model. Input: a state plus typed questions. Output types: `noul` (yes/no probability), `choice` (one of up to 255 options with a full distribution and confidence), `score` (rubric). No text generation. 64k-token budget. $0.042 per million input tokens, output free; 70–500ms. Stated weaknesses: negation, indirect reasoning, "context rot" with irrelevant state, option-order sensitivity. All benchmarks are self-reported; the 100–400× cost figures are against frontier models. Python SDK, Pydantic AI and Cloudflare integrations exist.

**Where it bears, specifically.** The judge is a closed classification over a slate. A `choice` over "c1..c5 or none" returns a candidate-bound verdict with a probability distribution, which is option (a) delivered by the output type, plus a judge-side confidence the band could threshold on. Latency of hundreds of milliseconds versus seconds matters on a synchronous write path. A 64k budget admits whole canonicals for a 5–10 slate; 20 × the longest canonical is ~58k tokens, borderline.

**Where it does not.** Retrieval: recall@k is an embedding and store question, untouched. The fixture gaps, the topic guard, ζ, and the measurement flaws (P3, P4) are ours. Its accuracy on this task is unmeasured, and its stated weaknesses (irrelevant context, indirect reasoning) are exactly what a slate of long memories presents; "amends vs restates" needs reading for a novel fragment across long text. Its option-order sensitivity undercuts "bound by construction": the verdict is bound to a position the model is sensitive to, on a slate whose target position already varies. Cost is not a discriminator against gpt-4o-mini: about 3.6× on input, both well under a cent per write.

**Honest verdict.** Relevant to one sub-problem, the judge's output contract and latency, with a real mechanism. Not a fix for the binding constraint as ruled. The same candidate binding is available today from gpt-4o-mini with a structured-output schema (option D, a day), and from a local cross-encoder (option H) with no vendor and no order sensitivity; Jev has to beat those, not the status quo. Testable for pennies with the existing eval harness once an API key exists, but only after C fixes the harness, otherwise it inherits the elision and slot-0 seeding. Early-access vendor on the write path is a dependency decision in its own right; fail-open already bounds an outage.

## 7. Record hygiene for Leo to direct (nothing done)

- 3169 metadata: "3135 blocked on 3133" is stale (3133 done); "4798 item 7" no longer resolves (coalesced into 5277 without it); the composite is any-attach, strict is ~0.16.
- tools.py DEFERRED comment: premise 1 false since 08-26.
- config.yaml `candidate_k` comment and its three echoes: one cluster, not the corpus.
- esc-3169-1 detail (write-once): "no before_done", "restart".
- Task 3134 `done_provenance.commit` points at an unrelated commit.
- Both escalation records: esc-3169-1 and the esc-3169-2 veto pin must be handled together; never `abandon` (a cancelled dep satisfies the scheduler and unblocks ε with the flag off). The recon queue holds a separate `esc-3169-1` (`reconciliation_stale_gate_backlog`) that is a different record.
