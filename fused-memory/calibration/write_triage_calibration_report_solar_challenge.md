# Write-triage calibration report

- `t_high`: `None`
- `t_low`: `None`
- deterministic-band false positives: **None**
- refusal reason: `empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=1596)`

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 0 | None | None | None | None | None |
| unrelated | 1416 | 0.18017136447318777 | 0.4378653158327845 | 0.5420488459184127 | 0.6122768768541348 | 0.8480328924734394 |
| hard_negative | 180 | 0.49376197909240227 | 0.6729637916349128 | 0.7826780678183197 | 0.8600855092795965 | 0.9978265624349195 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 0 | 0 | 0 |
| unrelated | 0 | 0 | 1416 |
| hard_negative | 0 | 0 | 180 |

## Per-category bands

| category | n true_dup | n negative | t_high | t_low | pooled t_high admits | reason |
|---|---|---|---|---|---|---|
| observations_and_summaries | 0 | 1035 | None | None | None | empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=1035) |
| preferences_and_norms | 0 | 6 | None | None | None | empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=6) |
| procedural_knowledge | 0 | 21 | None | None | None | empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=21) |

## Candidate-retrieval recall

| k | hits | total | recall |
|---|---|---|---|
| 1 | 0 | 43 | 0.0 |
| 3 | 4 | 43 | 0.09302325581395349 |
| 5 | 8 | 43 | 0.18604651162790697 |
| 10 | 29 | 43 | 0.6744186046511628 |
| 20 | 37 | 43 | 0.8604651162790697 |

Canonicals absent from the corpus (excluded from the denominator): 0

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration_solar_challenge.jsonl`
- `embedder_model`: `text-embedding-3-small`
- `embedder_dimensions`: `1536`
- `project_id`: `solar_challenge`
- `retrieval_mode`: `production`
- `retrieval_call`: `fused_memory.server.write_triage::retrieve_candidates`
- `canonical_aliases_path`: `None`
- `canonical_aliases_count`: `0`
- `record_count`: `57`
- `cluster_count`: `14`
- `per_category_record_counts`: `{'observations_and_summaries': 46, 'preferences_and_norms': 4, 'procedural_knowledge': 7}`
- `cross_category_dropped`: `534`
- `degraded_retrievals`: `0`
- `pair_counts`: `{'true_dup': 0, 'unrelated': 1416, 'hard_negative': 180}`
- `per_category_pair_counts`: `{'observations_and_summaries': {'true_dup': 0, 'unrelated': 860, 'hard_negative': 175}, 'preferences_and_norms': {'true_dup': 0, 'unrelated': 5, 'hard_negative': 1}, 'procedural_knowledge': {'true_dup': 0, 'unrelated': 17, 'hard_negative': 4}}`
