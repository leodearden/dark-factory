# Write-triage judge accuracy report

## Per-class accuracy

| class | n | correct | accuracy |
|---|---|---|---|
| duplicate | 75 | 40 | 0.5333 |
| distinct | 3 | 2 | 0.6667 |
| pseudo_contradiction | 6 | 5 | 0.8333 |
| distractor | 18 | 18 | 1.0 |

## Confusion — expected class by observed verdict

| class | stored | restated | amended | contested |
|---|---|---|---|---|
| duplicate | 31 | 10 | 30 | 4 |
| distinct | 2 | 0 | 1 | 0 |
| pseudo_contradiction | 3 | 1 | 1 | 1 |
| distractor | 18 | 0 | 0 | 0 |

## Duplicate attach split (a distribution, not an error term)

- `restated`: 10
- `amended`: 30

## Contested

- contested verdicts observed: **5**, all of which are FALSE POSITIVES.
- ground truth available: `False`
- `no_positive_contested_labels: the fixture carries 6 pseudo_contradiction records, every one curator-adjudicated NOT a contradiction, and 0 records labelled as a genuine contradiction. Contested recall and precision are therefore unmeasurable against this corpus; only the false-positive count below is a measurement.`

## Caveats

- No accuracy floor is asserted anywhere in this script or its tests (PRD D10). This artifact is evidence for a human decision, not a gate.
- false_contested counts EVERY contested verdict, and every one of them is a false positive — see contested_ground_truth. A judge structurally incapable of ever answering `contested` would score identically to a perfect one here, so a low number is not evidence that the contradiction detector works.
- The duplicate class accepts BOTH `restated` and `amended`, because the curator's labels do not separate a verbatim restatement from a rediscovery carrying a novel fragment. The split between them is reported as a distribution and is not scored as error.
- The distractor class is a control this script constructs, not a curator label: one case per cluster whose slate carries no correct attach target at all. It is what distinguishes a judge that classifies from a judge that attaches to whatever it is shown.

## Provenance

- `fixture_path`: `tests/fixtures/write_triage_calibration.jsonl`
- `judge_provider`: `openai`
- `judge_model`: `gpt-4o-mini`
- `limit`: `None`
- `record_count`: `104`
- `case_count`: `102`
- `candidate_count`: `5`
- `candidate_count_min`: `5`
- `distractor_count_requested`: `4`
- `distractor_count`: `4`
