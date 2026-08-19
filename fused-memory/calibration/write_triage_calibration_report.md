# Write-triage calibration report

- `t_high`: `0.8868282526657318`
- `t_low`: `0.5229672433064231`
- deterministic-band false positives: **0**

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 301 | 0.3044689234163344 | 0.6164660007110104 | 0.709713152372983 | 0.7847931720229323 | 0.96570744126053 |
| unrelated | 5037 | 0.1491529252563599 | 0.3639902745784614 | 0.4254931064390011 | 0.49766807005413743 | 0.8349988478146576 |
| hard_negative | 18 | 0.722523229400382 | 0.7481496080601845 | 0.7846337403069573 | 0.8215389446744906 | 0.8843433935965617 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 7 | 279 | 15 |
| unrelated | 0 | 920 | 4117 |
| hard_negative | 0 | 18 | 0 |

## Per-category bands

| category | n true_dup | n negative | t_high | t_low | pooled t_high admits | reason |
|---|---|---|---|---|---|---|
| observations_and_summaries | 4 | 51 | None | None | 0 | insufficient_pairs: 4 duplicate and 51 negative pair(s) measured, below the 20/20 minimum. A fresh pair falls outside a sample this small's observed extreme with expected probability 1/(n+1), and both band edges are fitted to exactly that extreme. Refusing a cutoff the sample size cannot support — this category runs on the disclosed pooled fallback instead. |
| preferences_and_norms | 28 | 0 | None | None | 0 | empty_class: cannot separate two classes when one is empty (n_duplicate=28, n_negative=0) |
| procedural_knowledge | 242 | 3328 | 0.839045608910388 | 0.5750178537519799 | 0 |  |

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
