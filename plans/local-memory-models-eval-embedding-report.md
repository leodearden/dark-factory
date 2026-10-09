# LME ι embedding-axis report (task 3722)

PRD-MARKER:local-memory-models-eval embedding-report

PRD `plans/local-memory-models-eval-prd.md`, task ι. The decision rule is
`plans/local-memory-models-eval-preregistration.md` §6, and §7 governs a negative. Every
number below is read from a committed file in `plans/local-memory-models-eval-embedding/`,
or from `harness.py embed-compare` over those files (its output is the appendix). The two
exceptions are the read-only census in §4 and §7 and the §7 backfill projection, which is
hand arithmetic over measured counts. The directory's `README.md` records how the evidence
was taken, including why the runs' code sha predates a rebase of this branch. Lines marked
*Hypothesis:* are inferences, not measurements.

## 1. Verdict

The rule: an arm is non-inferior when every with-indices margin admits it AND its
query-embed-latency-p95 is under the 15000 ms envelope with no latency query failed
(`embedding_preregistration.py::compare_embedding_arm`). A run whose re-embed left any
text unembedded is not judged at all; none did. embedding-only is reported, not
deciding.

| arm | with-indices margins | latency envelope | verdict |
|---|---|---|---|
| granite-embedding-english-r2 (768) | all admit | 30.1 ms, admits | **non-inferior** |
| qwen3-embedding-0.6b (1024) | all admit, known-item-recall@10 exactly at the bound | 31.1 ms, admits | **non-inferior** |
| qwen3-embedding-4b (2560) | known-item-recall@10 fails | 64.9 ms, admits | inferior |
| gte-modernbert-base (768) | known-item-recall@10 fails | 40.3 ms, admits | inferior |

- **granite-embedding-english-r2** scores above the incumbent on every gated metric in
  both configurations. With indices it retrieves 179 of 199 known items in the top 10
  against the incumbent's 175, and its mrr is 0.7176 against 0.6811.
- **qwen3-embedding-0.6b** passes with nothing to spare. Its with-indices
  known-item-recall@10 is 174/199, one item below the incumbent's 175. The margin is one
  item, the floor 1/199, so 174 is admitted and 173 would not have been. Its recall@5
  (168 against 163) and mrr (0.6911 against 0.6811) are above the incumbent.
- **qwen3-embedding-4b** and **gte-modernbert-base** each find 172/199 with indices in the
  top 10: three items below the incumbent, two beyond the margin.

Non-inferiority is what was pre-registered and what these runs test. This report claims
no superiority. The largest gap in either direction is 7 known items of 199 (granite's
with-indices recall@5), and no superiority test was pre-registered or run.

## 2. Results against the pre-registered margins

The two control runs re-embedded the same frozen graph with `text-embedding-3-small` and
returned identical values on every gated metric in both configurations. So σ = 0, every
proportion's margin is its floor (1/199, one item), and the mrr margin is 0, which means
"not below the incumbent", as prereg §6 accepts. The reference rows are the control pair's
mean, with the a − b spread in parentheses. Every margin comes from
`embedding-preregistration-inputs.json`, and every admits/fails from `embed-compare`.

### with-indices (decides)

| arm | known-item-recall@5 | known-item-recall@10 | mrr |
|---|---|---|---|
| incumbent-embed-a (control) | 0.8191 (163/199) | 0.8794 (175/199) | 0.6811 |
| incumbent-embed-b (control) | 0.8191 (163/199) | 0.8794 (175/199) | 0.6811 |
| reference: control-pair mean (spread a − b) | 0.8191 (0.0000) | 0.8794 (0.0000) | 0.6811 (0.0000) |
| margin = max(2σ, floor) | 0.0050 (1/199) | 0.0050 (1/199) | 0.0000 (no floor) |
| admits a value ≥ | 0.8141 | 0.8744 | 0.6811 |
| qwen3-embedding-0.6b | 0.8442 (168/199) admits | 0.8744 (174/199) admits | 0.6911 admits |
| granite-embedding-english-r2 | 0.8543 (170/199) admits | 0.8995 (179/199) admits | 0.7176 admits |
| qwen3-embedding-4b | 0.8442 (168/199) admits | 0.8643 (172/199) **fails** | 0.6825 admits |
| gte-modernbert-base | 0.8291 (165/199) admits | 0.8643 (172/199) **fails** | 0.6907 admits |

### embedding-only (reported)

| arm | known-item-recall@5 | known-item-recall@10 | mrr |
|---|---|---|---|
| incumbent-embed-a (control) | 0.8291 (165/199) | 0.8643 (172/199) | 0.7570 |
| incumbent-embed-b (control) | 0.8291 (165/199) | 0.8643 (172/199) | 0.7570 |
| reference: control-pair mean (spread a − b) | 0.8291 (0.0000) | 0.8643 (0.0000) | 0.7570 (0.0000) |
| margin = max(2σ, floor) | 0.0050 (1/199) | 0.0050 (1/199) | 0.0000 (no floor) |
| admits a value ≥ | 0.8241 | 0.8593 | 0.7570 |
| qwen3-embedding-0.6b | 0.8342 (166/199) admits | 0.8794 (175/199) admits | 0.7662 admits |
| granite-embedding-english-r2 | 0.8593 (171/199) admits | 0.8945 (178/199) admits | 0.8015 admits |
| qwen3-embedding-4b | 0.8291 (165/199) admits | 0.8442 (168/199) **fails** | 0.7627 admits |
| gte-modernbert-base | 0.8291 (165/199) admits | 0.8593 (171/199) admits | 0.7707 admits |

Every model, the incumbent included, has a lower mrr with indices than without. With
indices, recall@10 rises for four of the five models and falls by one item for
qwen3-embedding-0.6b. The pass/fail pattern is the same in both configurations with one
exception: gte-modernbert-base passes embedding-only exactly at the bound (171/199) and
fails with indices.

### Latency and the reported metrics

| arm | query-embed-latency-p95 (n = 500) | under 15000 ms | mem0-known-item-recall@5 | mem0-known-item-recall@10 | mem0-mrr |
|---|---|---|---|---|---|
| incumbent-embed-a | 218.7 ms | yes | 0.2619 (11/42) | 0.3095 (13/42) | 0.2405 |
| incumbent-embed-b | 263.5 ms | yes | 0.2619 (11/42) | 0.3095 (13/42) | 0.2405 |
| qwen3-embedding-0.6b | 31.1 ms | yes | 0.4048 (17/42) | 0.4524 (19/42) | 0.2816 |
| granite-embedding-english-r2 | 30.1 ms | yes | 0.4286 (18/42) | 0.4524 (19/42) | 0.2768 |
| qwen3-embedding-4b | 64.9 ms | yes | 0.4286 (18/42) | 0.4524 (19/42) | 0.3419 |
| gte-modernbert-base | 40.3 ms | yes | 0.3571 (15/42) | 0.4286 (18/42) | 0.2440 |

The bound is `queue.search_timeout_seconds` 30 s / 2.0 = 15000 ms (prereg §6).
`embed-preregister` records the worse control's p95, 263.5 ms, as the incumbent's, and
every local arm is more than two orders of magnitude under the bound. The Mem0 columns
are reported, not gated. Every candidate finds more of the 42 registry phrasings'
canonicals than the incumbent does, but n is 42 (§6).

## 3. Index configurations, proven per run

Each run took its own scratch graph through both configurations and proved each one
before probing it. The proof is the `index-configuration` check in every `run.json`: a
fulltext probe on the arm's graph that must return 0 rows with every index dropped
(embedding-only), and at least 1 row once `ensure_indices` has built them (with-indices).
Every run also re-proved its topology after re-embedding and again after both probes
(`reembed-integrity`: 1834 nodes and 3101 edges, the reference's). It checked the frozen
reference's hash before re-embedding (`frozen-reference-unchanged`).

| arm | scratch graph | embedding-only probe | with-indices probe | reembed-integrity ×2 | frozen-reference-unchanged |
|---|---|---|---|---|---|
| incumbent-embed-a | `evalmem_lme_emb_ctl_a` | 0 rows | 1 row | PASS | PASS |
| incumbent-embed-b | `evalmem_lme_emb_ctl_b` | 0 rows | 1 row | PASS | PASS |
| qwen3-embedding-0.6b | `evalmem_lme_emb_qwen3_embedding_0_6b` | 0 rows | 1 row | PASS | PASS |
| granite-embedding-english-r2 | `evalmem_lme_emb_granite_embedding_english_r2` | 0 rows | 1 row | PASS | PASS |
| qwen3-embedding-4b | `evalmem_lme_emb_qwen3_embedding_4b` | 0 rows | 1 row | PASS | PASS |
| gte-modernbert-base | `evalmem_lme_emb_gte_modernbert_base` | 0 rows | 1 row | PASS | PASS |

After the teardown the frozen reference re-hashed byte-identical to
`plans/local-memory-models-eval-controls/frozen-reference.json`.

## 4. The current-production confound, measured

The PRD's §Sketch and prereg §6 framed embedding-only as today's production. Leo's
esc-3722-3 ruling added dated, framing-only errata beneath both passages (commit
`be4a44d624`, 2026-10-08). They record that `dark_factory` has carried the full fulltext
set since index task 3708 landed on 2026-10-02, so with-indices is its current production
as well as its future one. Those errata are the authority and are not restated here.

The errata measured `dark_factory` only. ι measured the rest of the PRD §Hazards list on
2026-10-09 at about 11:10 UTC, read-only, via
`docker exec docker-falkordb-1 redis-cli GRAPH.RO_QUERY <g> "CALL db.indexes()"`. It also
measured the live project graphs outside that list:

| graph | on the §Hazards list | nodes | fulltext on `RELATES_TO.fact` | fulltext on `Entity.name` | production configuration |
|---|---|---|---|---|---|
| `dark_factory` | yes | 33379 | yes | yes | with-indices |
| `reify` | yes | 36444 | yes | yes | with-indices |
| `know_live` | yes | 3093 | yes | yes | with-indices |
| `solar_challenge_platform` | yes | 7756 | yes | yes | with-indices |
| `autopilot_video` | yes | 1810 | yes | yes | with-indices |
| `pump_web_ui` | yes | 117 | yes | yes | with-indices |
| `my_solar_challenge` | yes | no such graph | — | — | — |
| `probe_e1_master` | yes | 0 | no index at all | no index at all | none (empty) |
| `_probe` | yes | 0 | no index at all | no index at all | none (empty) |
| `solar_challenge` | no | 12689 | yes | yes | with-indices |
| `scoping_intake` | no | 917 | yes | yes | with-indices |
| `knowlive` | no | 0 | no index at all | no index at all | none (empty) |

No live graph that holds data still runs embedding-only. The graphs without indices are
empty, and `my_solar_challenge` does not exist. embedding-only is reported because prereg
§6 requires it, but on 2026-10-09 it is the production configuration of no populated
graph.

## 5. Normalisation

There are two guards, one at each end, and either alone makes the comparison
scale-free:

- **At the embed seam.** Every stored and every query vector passes through
  `fused_memory/arm_harness/normalization.py::unit_vector`, called from
  `arm_embedder.py`. That one function L2-normalises. It refuses a vector that is
  non-finite, zero, or of the wrong dimension, and the raw norm is recorded before
  normalising.
- **In both stores.** FalkorDB's `vec.cosineDistance` is a true cosine, scale-invariant
  on 4.18.0: `cosDist([3,4],[6,8]) = 0` and `cosDist([1,0],[30,0]) = 0` were measured,
  and `test_live_embedding_integration.py` re-measures a stored unit vector against its
  30× twin at distance 0. Every Qdrant replica was created with `Distance.COSINE`
  (`mem0_replica.py`).

Raw norms per arm, from each `run.json`'s `graph_reembed.raw_norms` and
`replica_reembed.raw_norms`:

| arm | graph vectors: median [min, max] (n = 2721) | replica vectors: median [min, max] (n = 31237) |
|---|---|---|
| incumbent-embed-a | 1.0000 [0.9994, 1.0007] | 1.0000 [0.9993, 1.0008] |
| incumbent-embed-b | 1.0000 [0.9994, 1.0007] | 1.0000 [0.9993, 1.0008] |
| qwen3-embedding-0.6b | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| granite-embedding-english-r2 | 30.4293 [29.8656, 31.0096] | 30.2802 [29.6551, 31.0775] |
| qwen3-embedding-4b | 1.0000 [1.0000, 1.0000] | 1.0000 [1.0000, 1.0000] |
| gte-modernbert-base | 37.6103 [35.3104, 40.9083] | 37.3970 [35.4974, 39.9644] |

granite and gte-modernbert return vectors of norm about 30 and 37, as α measured. The
other arms return unit vectors to within 0.001. Both stores already score by cosine, so
the explicit normalisation changes no ranking. It is a guard against a future dot-product
store, not a correction to these results.

## 6. The probe, and its limits

- **Known-item query.** A known item is a reference `Episodic` node cited by at least one
  `RELATES_TO` edge's `episodes`. 199 of the 200 qualify; one is cited by none. Its query
  is the first K words of the episode's content. A hit is any of the top 10 edges that
  `GraphitiBackend.search` returns whose `episodes` cites that node.
- **Why K = 10.** K is the median word count of the 500 real transcript queries
  (`probe_set.py::derive_query_words`, rounded half up). A probe as long as a real query
  exercises the BM25 leg as production does. K must also stay under graphiti's 128-term
  fulltext cutoff (`search_utils.MAX_QUERY_LENGTH`). FalkorDB's `build_fulltext_query`
  returns `''` once a query's terms plus its group ids reach that cutoff; the BM25 leg then
  goes dead, and with-indices silently becomes embedding-only. `derive_query_words`
  refuses such a K.
- **Mem0 probe.** Its known items are E1's registry phrasings whose canonical the snapshot
  holds: 42 phrasings over 14 topics. Its metrics (`mem0-known-item-recall@5/@10`,
  `mem0-mrr`) are reported, not gated, and n = 42 is small.
- **Transcript queries** drive `query-embed-latency-p95` only. They carry no relevance
  labels.
- **The cosine floor.** The edge cosine leg keeps only edges scoring
  `(2 − cosDist)/2 > 0.6`, that is cosine > 0.2 (graphiti's `DEFAULT_MIN_SCORE` as
  `sim_min_score`). That cutoff is carried over from production unchanged, and it was not
  tuned to any embedder's space in particular. An embedder whose unrelated texts sit at
  higher or lower cosine meets it differently.
- **Timeouts read as misses.** `GraphitiBackend.search` returns `[]` on its 30 s timeout,
  which is indistinguishable from a miss. No run recorded a query-embedding failure
  (`query_failures` is 0 everywhere), but a search timeout leaves no trace there.

## 7. Re-embed throughput and the full-backfill projection

`reembed-throughput` is vectors per embedding second over each run's 33958 texts: 2721
graph names and facts, plus the 31237 Mem0 snapshot points. It was measured at batch 64,
concurrency 4. Write seconds are shown separately and are not part of the rate.

The backfill volume was measured read-only on 2026-10-09 at about 11:10 UTC. It is 168700
graph vectors (`Entity` plus `RELATES_TO` across the eight populated graphs in §4) plus
80271 points across every `fused_*` Qdrant collection, 248971 in all.

| arm | reembed-throughput (vectors/s) | graph-only rate | replica-only rate | embedding s | write s | backfill, embedding (single rate) | backfill, embedding (split rate) | backfill, writes |
|---|---|---|---|---|---|---|---|---|
| incumbent-embed-a | 551.8 | 587.6 | 548.8 | 61.5 | 75.0 | 451 s (7.5 min) | 433 s (7.2 min) | 550 s (9.2 min) |
| incumbent-embed-b | 561.4 | 541.2 | 563.3 | 60.5 | 62.7 | 443 s (7.4 min) | 454 s (7.6 min) | 460 s (7.7 min) |
| qwen3-embedding-0.6b | 201.4 | 773.8 | 189.2 | 168.6 | 46.0 | 1236 s (20.6 min) | 642 s (10.7 min) | 338 s (5.6 min) |
| granite-embedding-english-r2 | 657.9 | 1354.5 | 629.7 | 51.6 | 30.1 | 378 s (6.3 min) | 252 s (4.2 min) | 221 s (3.7 min) |
| qwen3-embedding-4b | 35.7 | 226.3 | 33.3 | 950.8 | 122.6 | 6971 s (1.94 h) | 3158 s (52.6 min) | 899 s (15.0 min) |
| gte-modernbert-base | 583.3 | 1335.1 | 556.1 | 58.2 | 48.7 | 427 s (7.1 min) | 271 s (4.5 min) | 357 s (5.9 min) |

The single-rate column is the plan's formula: 248971 divided by `reembed-throughput`.
That rate is dominated by the Mem0 texts, 31237 of the 33958. The split-rate column
projects the graph vectors at the measured graph-only rate and the Qdrant points at the
replica-only rate. Graph names and facts embed 2 to 7 times faster than Mem0 texts on the
local arms, and two-thirds of the backfill is graph vectors, so the split figure is the
closer projection. The write column projects at each run's measured write rate.

On either projection, a full backfill to either non-inferior arm takes minutes, not
hours. *Hypothesis:* the Qwen arms' slower Mem0 rate is sequence-length cost on a decoder
backbone, since the Mem0 texts are far longer than the graph texts. It was not measured
separately.

## 8. Public-benchmark anchors

| arm | public figure | suite |
|---|---|---|
| text-embedding-3-small (incumbent) | 62.3 | MTEB, v1-era (OpenAI's 2024 announcement, per the PRD) |
| qwen3-embedding-0.6b | 70.70 | MTEB(eng, v2) |
| granite-embedding-english-r2 | 59.5 | IBM 6-benchmark composite (arXiv:2508.21085, Fig. 1) |
| qwen3-embedding-4b | 74.60 | MTEB(eng, v2) |
| gte-modernbert-base | 64.38 MTEB English, average of 56 tasks; BEIR 55.33, average of 15 | Alibaba-NLP/gte-modernbert-base model card (read 2026-10-09) |

These numbers are not comparable with one another:
- The incumbent's 62.3 is a v1-era MTEB score and is NOT comparable with the Qwen arms'
  MTEB(eng, v2) figures.
- gte-modernbert's 56-task English average is the v1 suite too.
- granite's figure is IBM's own composite.

No apples-to-apples retrieval column exists across all five. The public figures do not
predict this probe either: qwen3-embedding-4b, the highest MTEB(eng, v2) figure here, is
inferior on it, and granite, whose figure cannot be placed on that scale, is the
strongest. This
eval's known-item probe, run on this project's own memory, is the primary instrument. The
anchors are context only.

## 9. Qwen3-Embedding-4B residency (PRD Open Q5)

The condition did not fire: qwen3-embedding-4b is inferior under the pre-registered rule
(§1), so no production-residency estimate is given.

## Appendix: `embed-compare` output

Regenerate from `fused-memory/` with:

```bash
E=../plans/local-memory-models-eval-embedding
uv run --no-sync python scripts/local_memory_models_eval/harness.py embed-compare \
  --preregistration $E/embedding-preregistration-inputs.json \
  --run $E/runs/qwen3-embedding-0.6b/20261009T094148Z \
  --run $E/runs/granite-embedding-english-r2/20261009T095139Z \
  --run $E/runs/qwen3-embedding-4b/20261009T100100Z \
  --run $E/runs/gte-modernbert-base/20261009T102653Z
```

```
| arm | metric | index configuration | reference | margin | candidate | admits |
|---|---|---|---|---|---|---|
| qwen3-embedding-0.6b | known-item-recall@10 | embedding-only | 0.864321608040201 | 0.005025125628140704 | 0.8793969849246231 | True |
| qwen3-embedding-0.6b | known-item-recall@10 | with-indices | 0.8793969849246231 | 0.005025125628140704 | 0.8743718592964824 | True |
| qwen3-embedding-0.6b | known-item-recall@5 | embedding-only | 0.8291457286432161 | 0.005025125628140704 | 0.8341708542713567 | True |
| qwen3-embedding-0.6b | known-item-recall@5 | with-indices | 0.8190954773869347 | 0.005025125628140704 | 0.8442211055276382 | True |
| qwen3-embedding-0.6b | mrr | embedding-only | 0.7570252053920395 | 0.0 | 0.7662020419558109 | True |
| qwen3-embedding-0.6b | mrr | with-indices | 0.681064050410784 | 0.0 | 0.6910664433277498 | True |
| granite-embedding-english-r2 | known-item-recall@10 | embedding-only | 0.864321608040201 | 0.005025125628140704 | 0.8944723618090452 | True |
| granite-embedding-english-r2 | known-item-recall@10 | with-indices | 0.8793969849246231 | 0.005025125628140704 | 0.8994974874371859 | True |
| granite-embedding-english-r2 | known-item-recall@5 | embedding-only | 0.8291457286432161 | 0.005025125628140704 | 0.8592964824120602 | True |
| granite-embedding-english-r2 | known-item-recall@5 | with-indices | 0.8190954773869347 | 0.005025125628140704 | 0.8542713567839196 | True |
| granite-embedding-english-r2 | mrr | embedding-only | 0.7570252053920395 | 0.0 | 0.8015474196378719 | True |
| granite-embedding-english-r2 | mrr | with-indices | 0.681064050410784 | 0.0 | 0.7175580282364202 | True |
| qwen3-embedding-4b | known-item-recall@10 | embedding-only | 0.864321608040201 | 0.005025125628140704 | 0.8442211055276382 | False |
| qwen3-embedding-4b | known-item-recall@10 | with-indices | 0.8793969849246231 | 0.005025125628140704 | 0.864321608040201 | False |
| qwen3-embedding-4b | known-item-recall@5 | embedding-only | 0.8291457286432161 | 0.005025125628140704 | 0.8291457286432161 | True |
| qwen3-embedding-4b | known-item-recall@5 | with-indices | 0.8190954773869347 | 0.005025125628140704 | 0.8442211055276382 | True |
| qwen3-embedding-4b | mrr | embedding-only | 0.7570252053920395 | 0.0 | 0.7627303182579565 | True |
| qwen3-embedding-4b | mrr | with-indices | 0.681064050410784 | 0.0 | 0.6824818537130095 | True |
| gte-modernbert-base | known-item-recall@10 | embedding-only | 0.864321608040201 | 0.005025125628140704 | 0.8592964824120602 | True |
| gte-modernbert-base | known-item-recall@10 | with-indices | 0.8793969849246231 | 0.005025125628140704 | 0.864321608040201 | False |
| gte-modernbert-base | known-item-recall@5 | embedding-only | 0.8291457286432161 | 0.005025125628140704 | 0.8291457286432161 | True |
| gte-modernbert-base | known-item-recall@5 | with-indices | 0.8190954773869347 | 0.005025125628140704 | 0.8291457286432161 | True |
| gte-modernbert-base | mrr | embedding-only | 0.7570252053920395 | 0.0 | 0.7706708143894073 | True |
| gte-modernbert-base | mrr | with-indices | 0.681064050410784 | 0.0 | 0.6907035175879397 | True |
envelope qwen3-embedding-0.6b: p95 31.063311733305454 ms, 0 failed queries, bound 15000.0 ms, admits True
reported qwen3-embedding-0.6b mem0-known-item-recall@5: 0.40476190476190477 (n 42)
reported qwen3-embedding-0.6b mem0-known-item-recall@10: 0.4523809523809524 (n 42)
reported qwen3-embedding-0.6b mem0-mrr: 0.2816137566137566 (n 42)
reported qwen3-embedding-0.6b reembed-throughput: 201.3845934981401 (n 33958)
non_inferior qwen3-embedding-0.6b: True
envelope granite-embedding-english-r2: p95 30.112157110124826 ms, 0 failed queries, bound 15000.0 ms, admits True
reported granite-embedding-english-r2 mem0-known-item-recall@5: 0.42857142857142855 (n 42)
reported granite-embedding-english-r2 mem0-known-item-recall@10: 0.4523809523809524 (n 42)
reported granite-embedding-english-r2 mem0-mrr: 0.27681405895691613 (n 42)
reported granite-embedding-english-r2 reembed-throughput: 657.914016649956 (n 33958)
non_inferior granite-embedding-english-r2: True
envelope qwen3-embedding-4b: p95 64.88543702289462 ms, 0 failed queries, bound 15000.0 ms, admits True
reported qwen3-embedding-4b mem0-known-item-recall@5: 0.42857142857142855 (n 42)
reported qwen3-embedding-4b mem0-known-item-recall@10: 0.4523809523809524 (n 42)
reported qwen3-embedding-4b mem0-mrr: 0.3419312169312169 (n 42)
reported qwen3-embedding-4b reembed-throughput: 35.71350848204817 (n 33958)
non_inferior qwen3-embedding-4b: False
envelope gte-modernbert-base: p95 40.29964283108711 ms, 0 failed queries, bound 15000.0 ms, admits True
reported gte-modernbert-base mem0-known-item-recall@5: 0.35714285714285715 (n 42)
reported gte-modernbert-base mem0-known-item-recall@10: 0.42857142857142855 (n 42)
reported gte-modernbert-base mem0-mrr: 0.24404761904761904 (n 42)
reported gte-modernbert-base reembed-throughput: 583.3398339000194 (n 33958)
non_inferior gte-modernbert-base: False
```
