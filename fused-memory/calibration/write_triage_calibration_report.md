# Write-triage calibration report

- `t_high`: `0.8868282526657318`
- `t_low`: `0.52299575012145`
- deterministic-band false positives: **0**

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 301 | 0.30449224120515567 | 0.6163819354518215 | 0.7086608425629257 | 0.7847931720229323 | 0.9657190014923018 |
| unrelated | 5037 | 0.14917837344772372 | 0.3639029977781052 | 0.4254610259147245 | 0.4976647976107755 | 0.8349988478146576 |
| hard_negative | 18 | 0.722523229400382 | 0.748194001898148 | 0.7846622725948826 | 0.8216043288606502 | 0.8843433935965617 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 7 | 279 | 15 |
| unrelated | 0 | 917 | 4120 |
| hard_negative | 0 | 18 | 0 |

## Per-category bands

| category | n true_dup | n negative | t_high | t_low | pooled t_high admits | reason |
|---|---|---|---|---|---|---|
| observations_and_summaries | 4 | 51 | 0.8868282526657318 | 0.6030894616576438 | 0 |  |
| preferences_and_norms | 28 | 0 | None | None | 0 | empty_class: cannot separate two classes when one is empty (n_duplicate=28, n_negative=0) |
| procedural_knowledge | 242 | 3328 | 0.8390612912630236 | 0.5750151021898863 | 0 |  |

## Candidate-retrieval recall

| k | hits | total | recall |
|---|---|---|---|
| 1 | 17 | 63 | 0.2698412698412698 |
| 3 | 25 | 63 | 0.3968253968253968 |
| 5 | 32 | 63 | 0.5079365079365079 |
| 10 | 45 | 63 | 0.7142857142857143 |

Canonicals absent from the corpus (excluded from the denominator): 21

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration.jsonl`
- `project_id`: `reify`
- `embedder_model`: `text-embedding-3-small`
- `embedder_dimensions`: `1536`
- `search_stores`: `mem0 (MemoryService.search, stores=[mem0])`
- `search_categories`: `all`
- `record_count`: `104`
- `cluster_count`: `20`
- `per_category_record_counts`: `{'procedural_knowledge': 85, 'observations_and_summaries': 11, 'preferences_and_norms': 8}`
- `cross_category_dropped`: `1703`
- `pair_counts`: `{'true_dup': 301, 'unrelated': 5037, 'hard_negative': 18}`
- `per_category_pair_counts`: `{'procedural_knowledge': {'true_dup': 242, 'unrelated': 3316, 'hard_negative': 12}, 'observations_and_summaries': {'true_dup': 4, 'unrelated': 45, 'hard_negative': 6}, 'preferences_and_norms': {'true_dup': 28, 'unrelated': 0, 'hard_negative': 0}}`
