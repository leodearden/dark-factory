# Write-triage calibration report

- `t_high`: `0.8866152951292025`
- `t_low`: `0.5229672433064231`
- deterministic-band false positives: **0**

## Measured distributions

| pair class | n | min | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| true_dup | 301 | 0.304489836999617 | 0.616469997275294 | 0.7097019587340561 | 0.7848812749556767 | 0.9656990321321789 |
| unrelated | 5037 | 0.14919289635377908 | 0.363907026930481 | 0.4254843144420686 | 0.4976748718478162 | 0.8349946484778277 |
| hard_negative | 18 | 0.72269872821749 | 0.7481498721880528 | 0.7846241438547253 | 0.8215389446744906 | 0.8843398698463949 |

## Per-band counts

| pair class | deterministic (s>=t_high) | judge | store |
|---|---|---|---|
| true_dup | 7 | 279 | 15 |
| unrelated | 0 | 920 | 4117 |
| hard_negative | 0 | 18 | 0 |

## Candidate-retrieval recall

| k | hits | total | recall |
|---|---|---|---|
| 1 | 20 | 68 | 0.29411764705882354 |
| 3 | 29 | 68 | 0.4264705882352941 |
| 5 | 35 | 68 | 0.5147058823529411 |
| 10 | 50 | 68 | 0.7352941176470589 |

Canonicals absent from the corpus (excluded from the denominator): 16

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
