# Write-triage calibration report

- `t_high`: `0.8868293243724489`
- `t_low`: `0.522971590497168`
- deterministic-band false positives: **0**

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 301 | 0.3044689234163344 | 0.6165531913206045 | 0.7097019587340561 | 0.7848736212886502 | 0.9657105894445585 |
| unrelated | 5037 | 0.1491529252563599 | 0.3639867511771781 | 0.42545052030721703 | 0.49767549585999604 | 0.8349098187756331 |
| hard_negative | 18 | 0.722523229400382 | 0.7481461574125756 | 0.7846622725948826 | 0.8216043288606502 | 0.8843326154083471 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 7 | 279 | 15 |
| unrelated | 0 | 920 | 4117 |
| hard_negative | 0 | 18 | 0 |

## Candidate-retrieval recall

| k | hits | total | recall |
|---|---|---|---|
| 1 | 18 | 63 | 0.2857142857142857 |
| 3 | 26 | 63 | 0.4126984126984127 |
| 5 | 32 | 63 | 0.5079365079365079 |
| 10 | 46 | 63 | 0.7301587301587301 |

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
- `pair_counts`: `{'true_dup': 301, 'unrelated': 5037, 'hard_negative': 18}`
