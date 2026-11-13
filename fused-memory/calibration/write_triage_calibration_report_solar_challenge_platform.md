# Write-triage calibration report

- `t_high`: `None`
- `t_low`: `None`
- deterministic-band false positives: **None**
- refusal reason: `empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=2850)`

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 0 | None | None | None | None | None |
| unrelated | 2636 | 0.13766340271753896 | 0.3234685074548978 | 0.3855726736911569 | 0.4884922313607716 | 0.9001579636364982 |
| hard_negative | 214 | 0.4541831350264498 | 0.7262540147060806 | 0.7765010831273275 | 0.8357454548732174 | 0.9604052090895375 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 0 | 0 | 0 |
| unrelated | 0 | 0 | 2636 |
| hard_negative | 0 | 0 | 214 |

## Per-category bands

| category | n true_dup | n negative | t_high | t_low | pooled t_high admits | reason |
|---|---|---|---|---|---|---|
| observations_and_summaries | 0 | 561 | None | None | None | empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=561) |
| preferences_and_norms | 0 | 0 | None | None | None | empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=0) |
| procedural_knowledge | 0 | 820 | None | None | None | empty_class: cannot separate two classes when one is empty (n_duplicate=0, n_negative=820) |

## Candidate-retrieval recall

| k | hits | total | recall |
|---|---|---|---|
| 1 | 0 | 54 | 0.0 |
| 3 | 18 | 54 | 0.3333333333333333 |
| 5 | 32 | 54 | 0.5925925925925926 |
| 10 | 35 | 54 | 0.6481481481481481 |
| 20 | 41 | 54 | 0.7592592592592593 |

Canonicals absent from the corpus (excluded from the denominator): 0

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration_solar_challenge_platform.jsonl`
- `embedder_model`: `text-embedding-3-small`
- `embedder_dimensions`: `1536`
- `project_id`: `solar_challenge_platform`
- `retrieval_mode`: `production`
- `retrieval_call`: `fused_memory.server.write_triage::retrieve_candidates`
- `canonical_aliases_path`: `None`
- `canonical_aliases_count`: `0`
- `record_count`: `76`
- `cluster_count`: `22`
- `per_category_record_counts`: `{'observations_and_summaries': 34, 'procedural_knowledge': 41, 'preferences_and_norms': 1}`
- `cross_category_dropped`: `1469`
- `degraded_retrievals`: `0`
- `pair_counts`: `{'true_dup': 0, 'unrelated': 2636, 'hard_negative': 214}`
- `per_category_pair_counts`: `{'observations_and_summaries': {'true_dup': 0, 'unrelated': 465, 'hard_negative': 96}, 'procedural_knowledge': {'true_dup': 0, 'unrelated': 722, 'hard_negative': 98}, 'preferences_and_norms': {'true_dup': 0, 'unrelated': 0, 'hard_negative': 0}}`
