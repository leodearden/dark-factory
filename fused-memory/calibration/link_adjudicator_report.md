# Link adjudicator report — hand links (δm, task 6185)

Rendered from `link_adjudicator_report.json`, which is normative; this file only restates it.
Each cell is `value (num/den) [Wilson 95% low, high]`; a `None` value has denominator 0.

## Population

- `n_pairs` (scored): **359**
- `n_misfile`: 15 · `n_corrects`: 111 · `n_belongs`: 344 · `n_unclear`: 0
- `excluded`: 0

## Arms — H3 figures

| arm | misfile_recall | false_detach_rate | corrects_recall | false_corrects_rate | parse_failure_rate |
|---|---|---|---|---|---|
| opus | 0.8667 (13/15) [0.6212, 0.9626] | 0.0029 (1/344) [0.0005, 0.0163] | 0.8829 (98/111) [0.8099, 0.9303] | 0.0043 (1/233) [0.0008, 0.0239] | 0.0000 (0/359) [0.0000, 0.0106] |
| sonnet | 0.6667 (10/15) [0.4171, 0.8482] | 0.0494 (17/344) [0.0311, 0.0777] | 0.6667 (74/111) [0.5747, 0.7475] | 0.0258 (6/233) [0.0119, 0.0550] | 0.0000 (0/359) [0.0000, 0.0106] |

Definitions:

- `misfile_recall` = arm misfile / majority misfile (higher is better)
- `false_detach_rate` = arm misfile / majority belongs (lower is better)
- `corrects_recall` = arm CORRECTS / majority CORRECTS (higher is better)
- `false_corrects_rate` = arm CORRECTS / majority agreeing (lower is better)
- `parse_failure_rate` = pairs with no verdict / all pairs (lower is better)

## Arms — kind agreement (sanity metric, not a gate input)

| arm | kind_agreement | kind_agreement_always_amendment_baseline |
|---|---|---|
| opus | 0.9255 (149/161) [0.8742, 0.9569] | 0.8882 (143/161) [0.8302, 0.9281] |
| sonnet | 0.8696 (140/161) [0.8088, 0.9131] | 0.8882 (143/161) [0.8302, 0.9281] |

- `kind_agreement` = arm heal kind == majority heal kind, sightings and half-links
- `kind_agreement_always_amendment_baseline` = share an always-"amendment" arm would get

## Arms — cost and failure storms

| arm | cost_usd | failure_storm |
|---|---|---|
| opus | 7.3130 | none |
| sonnet | 2.7138 | none |

## Selection

Threshold-free rule: highest `misfile_recall`, then lowest `false_detach_rate`, then lowest `false_corrects_rate`, then lower cost. The gates alone judge bounds.

- selected arm: **opus**
- `selection.misfile_recall`: 0.8667 (13/15) [0.6212, 0.9626]
- `selection.false_detach_rate`: 0.0029 (1/344) [0.0005, 0.0163]
- `selection.corrects_recall`: 0.8829 (98/111) [0.8099, 0.9303]
- `selection.false_corrects_rate`: 0.0043 (1/233) [0.0008, 0.0239]
- `selection.parse_failure_rate`: 0.0000 (0/359) [0.0000, 0.0106]

## Provenance

- `brief_sha256`: `"7f161b0ed8b8cd5d24592a926068c00d075efdc2106375d65cfe1879bbebed11"`
- `corpus_sha256`: `"e8a2574c8cc4e8c1835189902e763c07e4987da1e66aae8edd020e80c9854b75"`
- `field_chars`: `4000`
- `mode`: `"hand_link"`
- `pairs_sha256`: `null`
- `shard_size`: `40`
- `text_key_basis`: `"texts read live by id through get_memory_by_id, kept only while they hash to the corpus row"`
- `text_keys`: `null`

## Run

- command, from the repo root: `uv run --project fused-memory python fused-memory/scripts/eval_link_adjudicator.py --corpus fused-memory/calibration/hand_link_verdicts.jsonl --arms opus,sonnet --config fused-memory/config/config.yaml --out fused-memory/calibration/link_adjudicator_report.json`. Task 6185 names `--corpus` and `--arms`; `--config` is passed explicitly and `--out` is the script's own default for this mode.
- base commit: `fc55c9c7c8`; run 2026-10-08 15:59:14Z → 16:48:14Z, wall 49m00s, exit 0
- phases: live population read 15:59:14Z → 16:04:11Z (~4m57s; 359 pairs × 2 reads by id on :8002); opus 16:04:11Z → 16:33:26Z (29m15s, 9 shards); sonnet 16:33:26Z → 16:48:09Z (14m43s, 9 shards)
- spend (CLI `total_cost_usd`, OAuth subscription quota): opus $7.3130, sonnet $2.7138, total $10.0268
- shard size (PRD §10 Q4): measured at `link_heal.shard_size` 40 only — 18 shards, 0 failed, 0 parse failures; per shard opus ~3.2 min / ~$0.81, sonnet ~1.6 min / ~$0.30. No other size was run.
