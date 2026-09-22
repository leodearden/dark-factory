# Write-triage judge accuracy report

## Per-class accuracy

| class | n | correct | accuracy |
|---|---|---|---|
| duplicate | 75 | 41 | 0.5467 |
| distinct | 3 | 1 | 0.3333 |
| pseudo_contradiction | 6 | 5 | 0.8333 |
| distractor | 18 | 18 | 1.0 |

## Confusion — expected class by observed verdict

| class | amended | contested | restated | stored |
|---|---|---|---|---|
| duplicate | 40 | 7 | 1 | 27 |
| distinct | 2 | 0 | 0 | 1 |
| pseudo_contradiction | 3 | 1 | 0 | 2 |
| distractor | 0 | 0 | 0 | 18 |

## Duplicate attach split (a distribution, not an error term)

- `restated`: 1
- `amended`: 40

## Contested

- contested verdicts observed: **8**, all of which are FALSE POSITIVES.
- ground truth available: `False`
- `no_positive_contested_labels: the fixture carries 6 pseudo_contradiction records, every one curator-adjudicated NOT a contradiction, and 0 records labelled as a genuine contradiction. Contested recall and precision are therefore unmeasurable against this corpus; only the false-positive count below is a measurement.`

## Caveats

- No accuracy floor is asserted anywhere in this script or its tests (PRD D10). This artifact is evidence for a human decision, not a gate.
- false_contested counts EVERY contested verdict, and every one of them is a false positive — see contested_ground_truth. A judge structurally incapable of ever answering `contested` would score identically to a perfect one here, so a low number is not evidence that the contradiction detector works.
- The duplicate class accepts BOTH `restated` and `amended`, because the curator's labels do not separate a verbatim restatement from a rediscovery carrying a novel fragment. The split between them is reported as a distribution and is not scored as error.
- The distractor class is a control this script constructs, not a curator label: one case per cluster whose slate carries no correct attach target at all. It is what distinguishes a judge that classifies from a judge that attaches to whatever it is shown.
- Every accuracy here is measured over the WHOLE labelled corpus, not over the [t_low, t_high) middle band the production judge is actually responsible for. `build_judge_cases` emits a case for every non-canonical record and `run_judge_eval` calls the judge on each one directly — `decide_band`, `t_high` and `t_low` never enter the picture, and the band decision handed to `judge_write` is SYNTHESIZED as a middle-band one. So these figures include records that in production are answered deterministically without the judge ever seeing them, and whether the middle band alone would score higher or lower is not measured here. Filtering the cases to the band would need real per-record similarities and is deliberately not done.

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration.jsonl`
- `judge_provider`: `openai`
- `judge_model`: `gpt-4o-mini`
- `limit`: `None`
- `judge_candidate_count`: `5`
- `judge_enabled`: `True`
- `record_count`: `104`
- `case_count`: `102`
- `candidate_count`: `5`
- `candidate_count_min`: `5`
- `distractor_count_requested`: `4`
- `distractor_count`: `4`
