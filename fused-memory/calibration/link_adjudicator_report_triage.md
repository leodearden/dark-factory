# Link adjudicator report — λ triage pairs (ζ, task 6190)

Rendered from `link_adjudicator_report_triage.json`, which is normative; this file only restates it.
Each cell is `value (num/den) [Wilson 95% low, high]`; a `None` value has denominator 0.

## Population

- `n_pairs` (scored): **1216**
- `n_misfile`: 532 · `n_corrects`: 129 · `n_belongs`: 684 · `n_unclear`: 0
- `excluded`: 0

## Arms — H3 figures

| arm | misfile_recall | false_detach_rate | corrects_recall | false_corrects_rate | parse_failure_rate |
|---|---|---|---|---|---|
| opus | 0.8571 (456/532) [0.8248, 0.8843] | 0.0190 (13/684) [0.0111, 0.0322] | 0.9070 (117/129) [0.8444, 0.9460] | 0.0198 (11/555) [0.0111, 0.0351] | 0.0000 (0/1216) [0.0000, 0.0031] |
| sonnet | 0.7876 (419/532) [0.7508, 0.8202] | 0.1579 (108/684) [0.1325, 0.1871] | 0.3798 (49/129) [0.3007, 0.4659] | 0.0288 (16/555) [0.0178, 0.0463] | 0.0000 (0/1216) [0.0000, 0.0031] |

Definitions:

- `misfile_recall` = arm misfile / majority misfile (higher is better)
- `false_detach_rate` = arm misfile / majority belongs (lower is better)
- `corrects_recall` = arm CORRECTS / majority CORRECTS (higher is better)
- `false_corrects_rate` = arm CORRECTS / majority agreeing (lower is better)
- `parse_failure_rate` = pairs with no verdict / all pairs (lower is better)

## Arms — cost and failure storms

| arm | cost_usd | failure_storm |
|---|---|---|
| opus | 20.6276 | none |
| sonnet | 6.1481 | none |

## Selection

Threshold-free rule: highest `misfile_recall`, then lowest `false_detach_rate`, then lowest `false_corrects_rate`, then lower cost. The gates alone judge bounds.

- selected arm: **opus**
- `selection.misfile_recall`: 0.8571 (456/532) [0.8248, 0.8843]
- `selection.false_detach_rate`: 0.0190 (13/684) [0.0111, 0.0322]
- `selection.corrects_recall`: 0.9070 (117/129) [0.8444, 0.9460]
- `selection.false_corrects_rate`: 0.0198 (11/555) [0.0111, 0.0351]
- `selection.parse_failure_rate`: 0.0000 (0/1216) [0.0000, 0.0031]

## Provenance

- `brief_sha256`: `"7f161b0ed8b8cd5d24592a926068c00d075efdc2106375d65cfe1879bbebed11"`
- `corpus_sha256`: `"ecd95fc0b97bee838118dca5458d8cb3a2e3291ddd06e6211d3ef408bbb37ee9"`
- `field_chars`: `4000`
- `mode`: `"triage"`
- `pairs_sha256`: `"3f51509a110cc53acc0f8468fe61ada30198faba77fc7fd8d28b5b11485549e8"`
- `shard_size`: `40`
- `text_key_basis`: `"the keys of task 6151's committed fused-memory/calibration/write_triage_pairs_to_rate.jsonl, built by scripts/run_write_triage_population_arms.py::build_pairs_to_rate"`
- `text_keys`: `{"child": "entry_text", "parent": "target_text"}`

## Run

- command, from the repo root: `uv run --project fused-memory python fused-memory/scripts/eval_link_adjudicator.py --corpus fused-memory/calibration/write_triage_pair_verdicts.jsonl --pairs fused-memory/calibration/write_triage_pairs_to_rate.jsonl --arms opus,sonnet --config fused-memory/config/config.yaml --out fused-memory/calibration/link_adjudicator_report_triage.json`. Task 6190 names `--corpus` and `--arms`; `--pairs` is required for λ mode (without it the script reads the corpus as hand links), `--config` is passed explicitly, and `--out` is the script's own default for this mode. Arms: opus (the arm δm's selection picked) plus sonnet.
- text-key check (task 6190 step 1): every row of `write_triage_pairs_to_rate.jsonl` carries exactly `entry_id`, `entry_text`, `target_id`, `target_text`, the names the reader uses; no reader fix was needed.
- exclusions: λ rated every one of the 1,216 pairs in π's pairs file (unrated 0, tied 0), and that file holds no ι seed or already-rated pair (`write_triage_population.json` `pairs_to_rate.excluded_already_rated` 0; `write_triage_pair_verdicts.summary.json` asserts none is in ι seed), so sampled-out and seed exclusions both count 0. π's judge-band sample (658 of 1,454 writes) happens upstream of the pairs file and is not counted here.
- base commit: `fc55c9c7c8`; run 2026-10-08 16:48:56Z → 19:02:16Z, wall 2h13m20s, exit 0
- phases: opus 16:49:02Z → 18:38:20Z (1h49m18s, 31 shards); sonnet 18:38:20Z → 19:02:13Z (23m53s, 31 shards)
- spend (CLI `total_cost_usd`, OAuth subscription quota): opus $20.6276, sonnet $6.1481, total $26.7757
- shard size: `link_heal.shard_size` 40 — 62 shards, 0 failed, 0 parse failures; per shard opus ~3.5 min / ~$0.67, sonnet ~0.8 min / ~$0.20.
