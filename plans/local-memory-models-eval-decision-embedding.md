# LME λ decision record — embedding axis (task 3723)

PRD-MARKER:local-memory-models-eval decision-embedding

**PRD:** `plans/local-memory-models-eval-prd.md` task λ. **Upstream:** ι (task 3722).
**Consumer:** μ (task 3725), the born-at-L2 operator gate. **Status:** the verdict in §3
is PROPOSED. Leo's ruling at μ is the decision. This record changes no config:
`fused-memory/config/config.yaml` still names `embedder.model: "text-embedding-3-small"`.

§1–2 are observation and §3 is the sole interpretation. §4–8 are caveats, limits, a named
follow-up and the open human questions. Every number is read from a committed file named
beside it, mostly `plans/local-memory-models-eval-embedding-report.md` (the ι report) and
the evidence directory `plans/local-memory-models-eval-embedding/`. Lines marked
*Hypothesis:* are inferences, not measurements.

## 1. The pre-registered rule, cited by SHA

The rule is `plans/local-memory-models-eval-preregistration.md` at
`be4a44d6243b58a3293d66eb73f0ba56fa206efd`. That is the `preregistration_sha` all four
candidate specs carry (`plans/local-memory-models-eval-embedding/specs/*.json`), and
Leo's esc-3722-3 ruling binds θ and ι runs to it. It differs from η's `865a017d79` only
by the framing-only erratum under §6 (11 inserted lines; see the LLM record §1).

- **§6** applies prereg §3's formula, floors and code (`derive_margins`) to ι's own
  incumbent-embedder control pair. The gated metrics are known-item-recall@5,
  known-item-recall@10 and mrr, all lower is worse, per index configuration. with-indices
  decides, embedding-only is reported, and the query-embed-latency-p95 envelope is
  `queue.search_timeout_seconds` 30 s / 2.0 = 15000 ms.
- **§7**: a clean negative is acceptable; no slate widening, margin relaxing or re-runs.
- **Availability decides among non-inferior arms.** That clause is the PRD's
  (§"Sketch of approach", paragraph "Pre-registration before candidate arms", which
  implements D2) and is written out in prereg §5 for the LLM axis. §6 inherits the shared
  formula and code but does not restate the clause.

As implemented in `fused_memory/arm_harness/embedding_preregistration.py::compare_embedding_arm`,
an arm is non-inferior iff every with-indices margin admits AND its
query-embed-latency-p95 is < 15000 ms with zero failed latency queries.

The two controls returned identical values on every gated metric, so σ = 0. Each
proportion's margin is therefore its floor (1/199, one known item) and the mrr margin is 0
(ι report §2; `embedding-preregistration-inputs.json`). With indices, an arm is admitted
iff R@5 ≥ 0.8141, R@10 ≥ 0.8744 and mrr ≥ 0.6811.

## 2. The measured comparison

with-indices, the deciding configuration (ι report §1–2 and its `embed-compare` appendix):

| arm (dims) | R@5 | R@10 | mrr | query p95 | verdict |
|---|---|---|---|---|---|
| granite-embedding-english-r2 (768) | 170/199 | 179/199 | 0.7176 | 30.1 ms | **non-inferior** |
| qwen3-embedding-0.6b (1024) | 168/199 | 174/199 (exactly at the bound) | 0.6911 | 31.1 ms | **non-inferior** |
| qwen3-embedding-4b (2560) | 168/199 | 172/199 **fails** | 0.6825 | 64.9 ms | inferior |
| gte-modernbert-base (768) | 165/199 | 172/199 **fails** | 0.6907 | 40.3 ms | inferior |
| incumbent reference, text-embedding-3-small (1536) | 163/199 | 175/199 | 0.6811 | 263.5 ms (worse control) | — |

- Each inferior arm fails on with-indices R@10 alone. Every latency envelope admits, at
  least two orders of magnitude under the bound.
- **embedding-only** (reported, not deciding) shows the same pattern, except that
  gte-modernbert-base passes there exactly at the bound (171/199; ι report §2).
- **The deciding configuration is today's production.** with-indices is the production
  configuration of every populated live graph, per ι's read-only census of 2026-10-09
  (ι report §4) and the esc-3722-3 erratum.
- **Provenance caveat.** All six runs carry code_sha `8bb8c8893b`. That is a pre-rebase
  task-branch commit, not an ancestor of `main`. It is retrievable through the local tag
  `eval/lme-iota-3722-runs`, and esc-3722-6 checked and accepted its tree-equivalence
  (status resolved, 2026-10-09T21:28Z). The verdicts above are the current committed judge
  over the unchanged committed run data, which is exactly what λ's premise test (§8)
  re-runs.

## 3. Proposed verdict (INTERPRETATION)

**Move fused-memory's embedder to a local model**, in both stores (Graphiti/FalkorDB and
Mem0/Qdrant) and on both the write and interactive query paths. Two arms are
non-inferior, and among non-inferior arms availability decides between incumbent and
local.

**The pre-registration ranks no non-inferior arm against another.** §5 says "no quality
ranking" for the LLM axis, §6 is silent, and no superiority test was run (ι report §1). So
choosing between the two admitted arms is outside the rule, and Leo rules it.

λ PROPOSES **granite-embedding-english-r2** (768d), with **qwen3-embedding-0.6b** (1024d)
as the admitted alternative. The grounds are all reported, non-gated evidence:
- granite admits every deciding metric with room to spare, and sits above the incumbent on
  all three in both configurations. qwen3-0.6b admits R@10 with zero slack: 174 against a
  bound of 174.
- Re-embed throughput is 657.9 against 201.4 vectors/s. The full-backfill embedding
  projection at the split rate is about 252 s against 642 s (ι report §7).
- granite needs no query-side instruct prefix; Qwen3-Embedding requires one (PRD
  §"Candidate slate").
- granite's vectors are smaller: 768 dims against 1024.

Production-residency VRAM for either arm, beside whisper-writer, was not measured.

## 4. External-anchor caveat

- The incumbent's 62.3 MTEB is a 2024 v1-era score. It is NOT comparable with the
  candidates' MTEB(eng, v2) figures (qwen3-0.6b 70.70, qwen3-4b 74.60).
- granite's 59.5 is IBM's own composite, and gte-modernbert-base's 64.38 is the v1 suite.
- No apples-to-apples column exists. The public figures also failed to predict this
  probe: qwen3-embedding-4b, with the highest v2 score, was inferior, and granite was the
  strongest.
- The replay-based known-item eval on this project's own memory is the primary
  instrument; public benchmarks are a sanity anchor only (PRD §"Candidate slate" caveat;
  ι report §8).

## 5. Scope and limits of the gain (reported)

- **What a local embedder removes.** The embedder runs on the interactive query path of
  both stores: ~414k lifetime `search` ops against ~63k memory writes (PRD §"Background").
  A local embedder removes OpenAI from query-time embedding. It does not remove OpenAI from
  the Graphiti write path while the LLM axis stays on the incumbent
  (`plans/local-memory-models-eval-decision-llm.md`). Full independence needs both axes
  (θ report §5).
- *Hypothesis (θ's, θ report §5):* the raw credit-exhaustion 429s on 2026-10-03/04 came
  from the OpenAI embedder. The error text does not name the raising client.
- **Cost is not a motive.** Embedding spend at $0.02 per 1M tokens is negligible (PRD
  §"Background").
- **The Mem0 probe is small.** n = 42, reported only (ι report §6).
- **The edge cosine floor is untuned.** cosine > 0.2 was carried over from production
  and is not tuned for any embedder's space (ι report §6).
- **granite's raw vector norms are about 30.** Both stores score by cosine, so this has no
  effect. A dot-product store would need `normalization.py::unit_vector` normalisation
  (ι report §5).

## 6. Follow-up PRD this verdict warrants

The **embedding backfill/migration PRD**. It is unfiled and is authored only after Leo's
ruling at μ. λ files nothing (PRD packaging decision, Leo 2026-08-05). Its scope inputs,
from PRD §"Out of scope", D7 and D11, and ι report §7:
- **Qdrant rebuild** of every `fused_*` collection: 80,271 points.
- **FalkorDB re-embed and vector-index rebuild**: 168,700 graph vectors across the eight
  populated graphs.
- **A `calibrate_write_triage` re-run**, because its `t_high`/`t_low` depend on the
  embedder's space (`fused-memory/scripts/calibrate_write_triage.py`).
- **The substrate** is `fused_memory/maintenance/reindex.py`, which already reads
  `embedder.providers.openai.api_url` from config (β's plumbing).
- **Cutover atomicity under mixed dims**: 1536 → 768 or 1024, since mixed dims break cosine
  (PRD D7).
- **A permanent serving unit**, because α's units were transient eval units (D11). The
  serving-stack choice is PRD Open Q1, and the unit must stay resident beside
  whisper-writer.
- **The measured backfill projection**: minutes, not hours, for either admitted arm (ι
  report §7).

If Leo rules to stay on the incumbent, nothing is warranted.

## 7. Open human decisions (surfaced, not decided)

- Which admitted arm to adopt: granite-embedding-english-r2 (λ's proposal) or
  qwen3-embedding-0.6b.
- Whether to cut over the embedder while the LLM stays on OpenAI, accepting partial
  independence.

## 8. Premise pins

Each test re-applies the committed rule to the committed evidence. If one fails, this
record is stale.

- `fused-memory/tests/arm_harness/test_lme_decision_premises.py::test_the_preregistered_rule_admits_exactly_granite_and_qwen3_0_6b`
- `fused-memory/tests/arm_harness/test_lme_decision_premises.py::test_each_inferior_embedder_fails_only_with_indices_recall_at_10_and_every_envelope_admits`
- `fused-memory/tests/arm_harness/test_lme_embedding_artifacts.py::test_the_preregistration_inputs_re_derive_from_the_control_runs`,
  together with that file's spec and run-integrity pins.

No conflation-rate metric is used (PRD D12).
