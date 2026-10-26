# Write-triage calibration report

- `t_high`: `None`
- `t_low`: `None`
- deterministic-band false positives: **None**
- refusal reason: `not_separable: the highest measured negative (0.9268408843200905) is at or above every measured duplicate (max=0.9217958509264798), so no measured value separates the classes. Refusing to interpolate a threshold the data does not support — this outcome is itself the calibration finding.`

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 49 | 0.6062306005178615 | 0.6890775730058293 | 0.7575111086083243 | 0.8151815449799091 | 0.9217958509264798 |
| unrelated | 5359 | 0.11653607694274686 | 0.31776004546169756 | 0.38608712555746844 | 0.45616958234382227 | 0.8808805140029784 |
| hard_negative | 478 | 0.2868297144774432 | 0.5960853862189479 | 0.6683184443517652 | 0.7403295155698358 | 0.9268408843200905 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 0 | 0 | 49 |
| unrelated | 0 | 0 | 5359 |
| hard_negative | 0 | 0 | 478 |

## Per-category bands

| category | n true_dup | n negative | t_high | t_low | pooled t_high admits | reason |
|---|---|---|---|---|---|---|
| observations_and_summaries | 39 | 951 | 0.8578402160989046 | 0.6310130700781506 | None |  |
| preferences_and_norms | 10 | 2006 | None | None | None | not_separable: the highest measured negative (0.9268408843200905) is at or above every measured duplicate (max=0.9217958509264798), so no measured value separates the classes. Refusing to interpolate a threshold the data does not support — this outcome is itself the calibration finding. |

## Candidate-retrieval recall

| k | hits | total | recall |
|---|---|---|---|
| 1 | 0 | 91 | 0.0 |
| 3 | 10 | 91 | 0.10989010989010989 |
| 5 | 23 | 91 | 0.25274725274725274 |
| 10 | 36 | 91 | 0.3956043956043956 |
| 20 | 61 | 91 | 0.6703296703296703 |

Canonicals absent from the corpus (excluded from the denominator): 0

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration_dark_factory.jsonl`
- `embedder_model`: `text-embedding-3-small`
- `embedder_dimensions`: `1536`
- `project_id`: `dark_factory`
- `retrieval_mode`: `production`
- `retrieval_call`: `fused_memory.server.write_triage::retrieve_candidates`
- `canonical_aliases_path`: `None`
- `canonical_aliases_count`: `0`
- `record_count`: `109`
- `cluster_count`: `18`
- `per_category_record_counts`: `{'preferences_and_norms': 64, 'observations_and_summaries': 45}`
- `cross_category_dropped`: `2880`
- `degraded_retrievals`: `0`
- `pair_counts`: `{'true_dup': 49, 'unrelated': 5359, 'hard_negative': 478}`
- `per_category_pair_counts`: `{'preferences_and_norms': {'true_dup': 10, 'unrelated': 1629, 'hard_negative': 377}, 'observations_and_summaries': {'true_dup': 39, 'unrelated': 871, 'hard_negative': 80}}`
