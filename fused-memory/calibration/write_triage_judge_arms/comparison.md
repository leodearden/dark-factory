# Write-triage judge arms: prompt by model

Baseline arm: `shipped` x `gpt-6.1-sol`. Every arm judged the same population:

- `fixture_path`: `tests/fixtures/write_triage_calibration.jsonl`
- `slate_mode`: `retrieved`
- `project_id`: `reify`
- `t_high`: `0.8868282526657318`
- `t_low`: `0.5229672433064231`
- `candidate_k`: `20`
- `judge_candidate_count`: `5`
- `field_chars`: `4000`
- `case_count`: `84`
- `canonical_aliases_path`: `tests/fixtures/write_triage_calibration.canonical_aliases.json`

## Arms

| wording | model | cosine middle band | judged restate band | every judged case | false contested | pseudo_contradiction contested | duplicate contested |
|---|---|---|---|---|---|---|---|
| shipped | gpt-6.1-sol | 33/73 = `0.4521` | 7/11 = `0.6364` | 40/84 = `0.4762` | 38 | 5 | 33 |
| shipped | gpt-6.2-sol | 39/73 = `0.5342` | 6/11 = `0.5455` | 45/84 = `0.5357` | 34 | 5 | 29 |
| psi | gpt-6.1-sol | 34/73 = `0.4658` | 6/11 = `0.5455` | 40/84 = `0.4762` | 38 | 5 | 33 |
| psi | gpt-6.2-sol | 40/73 = `0.5479` | 7/11 = `0.6364` | 47/84 = `0.5595` | 32 | 5 | 27 |

## Prompt effect at each model (arm minus the `shipped` arm)

| model | wording | middle_band_accuracy | restate_band_accuracy | judged_accuracy | false_contested | pseudo_contradiction_contested | duplicate_contested |
|---|---|---|---|---|---|---|---|
| gpt-6.1-sol | psi | 0.0137 | -0.0909 | 0.0 | 0 | 0 | 0 |
| gpt-6.2-sol | psi | 0.0137 | 0.0909 | 0.0238 | -2 | 0 | -2 |

## Model effect at each prompt (arm minus the `gpt-6.1-sol` arm)

| wording | model | middle_band_accuracy | restate_band_accuracy | judged_accuracy | false_contested | pseudo_contradiction_contested | duplicate_contested |
|---|---|---|---|---|---|---|---|
| shipped | gpt-6.2-sol | 0.0821 | -0.0909 | 0.0595 | -4 | 0 | -4 |
| psi | gpt-6.2-sol | 0.0821 | 0.0909 | 0.0833 | -6 | 0 | -6 |
