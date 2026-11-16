# Duplicate-detector recall on the task 3130 fixture

Rendered from the JSON report beside this file by `scripts/measure_duplicate_detector_recall.py`.

**Setting:** fixture_only, 104 records, per-category top_k=10, embedder text-embedding-3-small (1536 dimensions). Pairs: 301 true, 18 hard negative, 5037 unrelated.

**Generator ceiling:** 262/301 (0.870) true pairs generated [0.828, 0.904]; 18 hard negative(s) generated; 27 cross-category true pair(s); 655 candidates.

**Chosen:** bge-reranker-v2-m3+claims — on point estimates: fail; on lower bounds: fail

| detector | class | granularity | status | skip reason | threshold | recall | recall 95% | precision | precision 95% | hard negatives flagged | AUC | AUC 95% | cost USD | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calibrated-ann | baseline | full_text | measured | n/a | n/a | 27/301 (0.090) | [0.062, 0.127] | 27/27 (1.000) | [0.875, 1.000] | 0/18 (0.000) | 0.2540 | [0.120, 0.387] | n/a | n/a |
| qwen3-reranker-0.6b | local_cross_encoder | full_text | measured | n/a | n/a | 0/301 (0.000) | [0.000, 0.013] | n/a | n/a | 0/18 (0.000) | 0.2030 | [0.078, 0.328] | 0.0000 | cuda:0 (NVIDIA GeForce RTX 3090) |
| mxbai-rerank-base-v2 | local_cross_encoder | full_text | measured | n/a | 0.999945 | 105/301 (0.349) | [0.297, 0.404] | 105/115 (0.913) | [0.847, 0.952] | 8/18 (0.444) | 0.3354 | [0.195, 0.476] | 0.0000 | cuda:0 (NVIDIA GeForce RTX 3090) |
| bge-reranker-v2-m3 | local_cross_encoder | full_text | measured | n/a | 0.889818 | 52/301 (0.173) | [0.134, 0.220] | 52/57 (0.912) | [0.811, 0.962] | 1/18 (0.056) | 0.4308 | [0.289, 0.572] | 0.0000 | cuda:0 (NVIDIA GeForce RTX 3090) |
| gpt-4o-mini-pairwise | llm_pairwise | full_text | measured | n/a | 0.00964212 | 145/301 (0.482) | [0.426, 0.538] | 145/161 (0.901) | [0.845, 0.938] | 10/18 (0.556) | 0.4035 | [0.262, 0.545] | 0.2516 | remote:api.openai.com |
| jina-reranker | hosted_api | full_text | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| voyage-rerank | hosted_api | full_text | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cohere-rerank | hosted_api | full_text | skipped | no_credential | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| qwen3-reranker-0.6b+claims | local_cross_encoder | claims | measured | n/a | 8.11458 | 9/301 (0.030) | [0.016, 0.056] | 9/10 (0.900) | [0.596, 0.982] | 1/18 (0.056) | 0.2901 | [0.153, 0.428] | 0.0000 | cuda:0 (NVIDIA GeForce RTX 3090) |
| mxbai-rerank-base-v2+claims | local_cross_encoder | claims | measured | n/a | 0.999273 | 12/301 (0.040) | [0.023, 0.068] | 12/12 (1.000) | [0.757, 1.000] | 0/18 (0.000) | 0.4747 | [0.335, 0.614] | 0.0000 | cuda:0 (NVIDIA GeForce RTX 3090) |
| bge-reranker-v2-m3+claims | local_cross_encoder | claims | measured | n/a | 0.368648 | 148/301 (0.492) | [0.436, 0.548] | 148/164 (0.902) | [0.847, 0.939] | 11/18 (0.611) | 0.3402 | [0.199, 0.481] | 0.0000 | cuda:0 (NVIDIA GeForce RTX 3090) |

## Recall at precision floors

| detector | 0.80 | 0.90 | 0.95 |
|---|---|---|---|
| qwen3-reranker-0.6b | 195/301 @ 6.5 | 0/301 @ n/a | 0/301 @ n/a |
| mxbai-rerank-base-v2 | 232/301 @ 0.999662 | 105/301 @ 0.999945 | 68/301 @ 0.999965 |
| bge-reranker-v2-m3 | 92/301 @ 0.806954 | 52/301 @ 0.889818 | 41/301 @ 0.912336 |
| gpt-4o-mini-pairwise | 209/301 @ 4.15917e-07 | 145/301 @ 0.00964212 | 42/301 @ 0.660244 |
| qwen3-reranker-0.6b+claims | 221/301 @ 1.50313 | 9/301 @ 8.11458 | 5/301 @ 8.38914 |
| mxbai-rerank-base-v2+claims | 143/301 @ 0.931305 | 12/301 @ 0.999273 | 12/301 @ 0.999273 |
| bge-reranker-v2-m3+claims | 211/301 @ 0.142394 | 148/301 @ 0.368648 | 17/301 @ 0.669307 |

## Gate proposal (pre-registered 2026-11-16, unratified)

- precision floor at the operating point: 0.90
- minimum paraphrase-band recall: 0.75
- minimum AUC (true vs hard negative): 0.80

Basis:

- precision 0.90: a filed gate's member list is the curator's working list (task 4916), and a false member is a distinct claim proposed for folding; at the fixture's mean cluster of about 5 members (104 records / 20 canonicals, about 10 pairs) that is about one false pair per gate.
- recall 0.75: guards against the systematically under-inclusive gate (the esc-5541 false-closure shape flagged in the 2026-07-30 review of PRD leaf kappa); transitive closure recovers a member through any one caught pair, so member recall is at least pair recall; the floor sits below the per-category top-10 generator's ceiling (262/301 = 0.870 when registered; this report's candidate_generation block re-measures it), so it is attainable by an adjudicator over this generator.
- AUC 0.80: Hosmer, Lemeshow and Sturdivant, Applied Logistic Regression, 3rd ed., section 5.2.4, rate 0.8-0.9 'excellent' discrimination; hard negatives are exactly the same-topic, different-claim pairs whose folding loses information; with 18 hard negatives the Hanley-McNeil interval is about +/-0.08, so both the point and lower-bound verdicts are reported for the ratifier (task 10244).

## Caveats

- The scan set is the fixture alone (fixture_only, 104 records): the most favourable setting, with no live-corpus distractors competing for each record's top_k=10 window.
- 27 of the 301 true pairs are cross-category. The per-category generator never forms them, so no adjudicator over it can catch them; recall still divides by all 301.
- Each arm's operating threshold is selected in-sample on these 104 records at the 0.90 precision floor; nothing is held out.
- AUC is rank-based (Mann-Whitney, ties count half) over every true and hard-negative pair, including pairs the generator never forms; its interval is Hanley-McNeil (1982).
- A pair's adjudicator score is the mean of its two directional scores: the scorers are query/document asymmetric, the duplicate relation is symmetric.
- The calibrated-ann baseline replays the 2026-08-11 detector (ANN hits at each category's calibrated cutoff, AUC over raw cosine). It is never eligible to be chosen.
- jev-choice is excluded by design: its scores are a probability distribution over one slate, not a pairwise judgement.
- The LLM pairwise arm is not run at claim granularity: claims multiply its call count roughly a hundredfold.
- pairs_over_max_length is a lower bound on truncation (entry plus candidate tokens, without the template): qwen3-reranker-0.6b at 8192 tokens: 0 slate pair(s) over; mxbai-rerank-base-v2 at 8192 tokens: 0 slate pair(s) over; bge-reranker-v2-m3 at 1024 tokens: 728 slate pair(s) over; qwen3-reranker-0.6b+claims at 8192 tokens: 0 slate pair(s) over; mxbai-rerank-base-v2+claims at 8192 tokens: 0 slate pair(s) over; bge-reranker-v2-m3+claims at 1024 tokens: 0 slate pair(s) over.
- jina-reranker was skipped (no_credential): JINA_API_KEY unset
- voyage-rerank was skipped (no_credential): VOYAGE_API_KEY unset
- cohere-rerank was skipped (no_credential): COHERE_API_KEY / CO_API_KEY unset

## Provenance

- `fixture_path`: "tests/fixtures/write_triage_calibration.jsonl"
- `record_count`: 104
- `top_k`: 10
- `ann_cutoffs`: {"observations_and_summaries": 0.8868282526657318, "preferences_and_norms": 0.8868282526657318, "procedural_knowledge": 0.839045608910388}
- `ann_cutoff_disclosure`: {"ann_disabled_uncalibrated": 0, "ann_pooled_fallback_categories": 2}
- `project_id`: null
- `measured_at`: "2026-11-16"
- `arm_filter`: null
- `device`: "auto"
- `local_batch_size`: 4
- `vram_cap_gib`: 8.0
- `pairwise_concurrency`: 20
- `max_arm_spend_usd`: 2.0
- `libraries`: {"torch": "2.14.0", "sentence-transformers": "6.1.0", "transformers": "5.17.0", "openai": "2.31.0"}
